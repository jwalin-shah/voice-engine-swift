#!/usr/bin/env python3
"""Debug: show the schema and first row of the parquet file."""
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

local = hf_hub_download("librispeech_asr", filename="clean/test/0000.parquet", repo_type="dataset")
table = pq.read_table(local, columns=["audio", "text", "speaker_id", "chapter_id", "id"])
print("Schema:", table.schema)
print("Num rows:", len(table))
row = table.slice(0, 1)
for col in table.schema.names:
    val = row[col][0].as_py()
    if col == "audio":
        print(f"\n{col} type={type(val)} value={str(val)[:200]}")
    else:
        print(f"{col}: {val}")
