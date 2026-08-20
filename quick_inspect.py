#!/usr/bin/env python3
"""Quick dataset inspection with smaller sample"""
from datasets import load_dataset
import json

print("Loading MSMARCO-XI dataset (this may take a few minutes)...")
ds = load_dataset("ai4bharat/MSMARCO-XI")

print("\n" + "="*80)
print("DATASET STRUCTURE")
print("="*80)

result = {
    "splits": list(ds.keys()),
    "info": {}
}

for split_name in ds.keys():
    split = ds[split_name]
    print(f"\nSplit: {split_name}")
    print(f"  Size: {len(split)}")
    print(f"  Columns: {split.column_names}")

    result["info"][split_name] = {
        "size": len(split),
        "columns": split.column_names,
        "features": {k: str(v) for k, v in split.features.items()}
    }

    # Sample first record
    if len(split) > 0:
        sample = split[0]
        print(f"  Sample record keys: {list(sample.keys())}")
        result["info"][split_name]["sample_keys"] = list(sample.keys())

        # Show first few values
        for k, v in list(sample.items())[:10]:
            if isinstance(v, str):
                print(f"    {k}: {v[:100]}{'...' if len(v) > 100 else ''}")
            elif isinstance(v, list):
                print(f"    {k}: list of {len(v)} items")
            else:
                print(f"    {k}: {type(v).__name__}")

with open("dataset_schema.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n" + "="*80)
print("Schema saved to dataset_schema.json")
print("="*80)
