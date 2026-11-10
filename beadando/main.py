#!/usr/bin/env python3

import pandas as pd


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

#convert to float
df = df.astype(float)


print(df.head())