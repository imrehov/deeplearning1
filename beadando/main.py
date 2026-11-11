#!/usr/bin/env python3

import tempfile

from preprocessing import preprocess


with tempfile.TemporaryDirectory() as temp_dir:
    preprocess("train_test_network.csv", temp_dir)
    