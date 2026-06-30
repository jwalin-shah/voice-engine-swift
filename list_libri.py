#!/usr/bin/env python3
"""List available parquet files in librispeech_asr dataset."""
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("librispeech_asr", repo_type="dataset")
parquet = sorted(f for f in files if f.endswith(".parquet"))
for pf in parquet:
    print(pf)
