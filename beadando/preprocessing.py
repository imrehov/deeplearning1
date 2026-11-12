#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path

def preprocess(input_file, output_dir):
    df = pd.read_csv(input_file)

    #dropping useless stuff
    df = df.drop(columns=["src_ip", "dst_ip", "dns_query", "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer", "http_trans_depth", "http_method",
                    "http_uri", "http_version", "http_request_body_len", "http_response_body_len", "http_status_code", "http_user_agent", "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice", "type"])

    #separate dataset into train, validate, test datasets, 70-15-15 split
    X = df.drop(columns=["label"])
    y = df["label"]

    #first split it 70-30
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        #this next option makes sure the label ratio stays relatively consistent
        stratify=y
    )

    #then split the 30 to 15-15
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    #normalization
    cols_to_normalize = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
    for c in cols_to_normalize:
        #step1: log1p
        X_train[c] = np.log1p(X_train[c])
        X_test[c] = np.log1p(X_test[c])
        X_val[c] = np.log1p(X_val[c])

    #step2: standardscaler
    #it can only be used on numeric cols
    #this will examine the cols in the dataset, calculates means and sclaes and saves it to the scaler object
    #important that we calc the data for the train set and use the same scaler for transforms on all other sets too
    scaler = StandardScaler().fit(X_train[cols_to_normalize])

    #transform data with the previously calced means and scales: z = (x-u)/s
    for dataset in (X_train, X_val, X_test):
        dataset[cols_to_normalize] = scaler.transform(dataset[cols_to_normalize])


    #creating buckets for ports, one-hot encoding resulted in 28k columns
    #will use 3 buckets:
    # 0–1023      system/well-known
    # 1024–49151  registered
    # 49152+      ephemeral
    port_bins = [-1, 1023, 49151, 65535]
    port_labels = ["well_known", "registered", "ephemeral"]
    for dataset in (X_train, X_val, X_test):
        dataset["src_port_bucket"] = pd.cut(
            dataset["src_port"],
            bins=port_bins,
            labels=port_labels
        )

    for dataset in (X_train, X_val, X_test):
        dataset["dst_port_bucket"] = pd.cut(
            dataset["dst_port"],
            bins=port_bins,
            labels=port_labels
        )

    for dataset in (X_train, X_val, X_test):
        dataset.drop(columns=["src_port", "dst_port"], inplace=True)

    #one-hot encoding stuff
    categorical_cols = ["proto", "conn_state", "service", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "src_port_bucket", "dst_port_bucket"]

    #encoder is fitted on the train dataset same as scaler
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    train_encoded = encoder.transform(X_train[categorical_cols])
    val_encoded = encoder.transform(X_val[categorical_cols])
    test_encoded = encoder.transform(X_test[categorical_cols])

    #get the names of the encoded cols to merge df
    encoded_names = encoder.get_feature_names_out(categorical_cols)

    #create dfs
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_names, index=X_train.index)
    val_encoded_df = pd.DataFrame(val_encoded, columns=encoded_names, index=X_val.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_names, index=X_test.index)

    X_train = X_train.drop(columns=categorical_cols)
    X_val = X_val.drop(columns=categorical_cols)
    X_test = X_test.drop(columns=categorical_cols)

    X_train = pd.concat([X_train, train_encoded_df], axis=1)
    X_val = pd.concat([X_val, val_encoded_df], axis=1)
    X_test = pd.concat([X_test, test_encoded_df], axis=1)

    #convert to float
    X_train = X_train.astype(float)
    X_val = X_val.astype(float)
    X_test = X_test.astype(float)

    #put y back, this way it will always be the last col -> easier in cntk
    train_df = X_train.copy()
    train_df["label"] = y_train

    val_df = X_val.copy()
    val_df["label"] = y_val

    test_df = X_test.copy()
    test_df["label"] = y_test

    # train_df.to_csv("train.csv", index=False, header=False)
    # val_df.to_csv("validate.csv", index=False, header=False)
    # test_df.to_csv("test.csv", index=False, header=False)

    output_dir = Path(output_dir)

    train_df.to_csv(output_dir / "train.csv", index=False, header=False)
    val_df.to_csv(output_dir / "validate.csv", index=False, header=False)
    test_df.to_csv(output_dir / "test.csv", index=False, header=False)

    # print(train_df.shape)
    # print(val_df.shape)
    # print(test_df.shape)