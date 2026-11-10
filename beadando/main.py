#!/usr/bin/env python3

import pandas as pd
import numpy as np


df = pd.read_csv("train_test_network.csv")

#dropping useless stuff
df = df.drop(columns=["src_ip", "dst_ip", "dns_query", "ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established", "ssl_subject", "ssl_issuer", "http_trans_depth", "http_method",
                "http_uri", "http_version", "http_request_body_len", "http_response_body_len", "http_status_code", "http_user_agent", "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice", "type"])

#one-hot encoding stuff
df = pd.get_dummies(
    df,
    columns=["proto", "conn_state", "service", "dns_qclass", "dns_qtype", "dns_rcode", "dns_AA", "dns_RD", "dns_RA", "dns_rejected"],
    drop_first=True
)

#creating buckets for ports, one-hot encoding resulted in 28k columns
#will use 3 buckets:
# 0–1023      system/well-known
# 1024–49151  registered
# 49152+      ephemeral
port_bins = [-1, 1023, 49151, 65535]
port_labels = ["well_known", "registered", "ephemeral"]

df["src_port_bucket"] = pd.cut(
    df["src_port"],
    bins=port_bins,
    labels=port_labels
)

df["dst_port_bucket"] = pd.cut(
    df["dst_port"],
    bins=port_bins,
    labels=port_labels
)

df.drop(columns=["src_port", "dst_port"], inplace=True)

# 6 cols after ohe
df = pd.get_dummies(
    df,
    columns=["src_port_bucket", "dst_port_bucket"],
    dtype=int
)

#convert to float
df = df.astype(float)

#normalization
cols_to_normalize = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
for c in cols_to_normalize:
    #step1: log1p
    df[c] = np.log1p(df[c])
    #step2: standardscalar
    

print(df.head())