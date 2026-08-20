#!/usr/bin/env python3
"""
Inspect MSMARCO-XI dataset to understand its structure.
DO NOT assume the schema - discover it programmatically.
"""

from datasets import load_dataset
import json

def inspect_dataset():
    print("=" * 80)
    print("INSPECTING MSMARCO-XI DATASET")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset from HuggingFace...")
    dataset = load_dataset("ai4bharat/MSMARCO-XI", trust_remote_code=True)

    # 1. Available splits
    print("\n" + "=" * 80)
    print("1. AVAILABLE SPLITS")
    print("=" * 80)
    print(f"Splits: {list(dataset.keys())}")

    # 2. For each split, inspect columns and size
    for split_name, split_data in dataset.items():
        print(f"\n{'=' * 80}")
        print(f"SPLIT: {split_name}")
        print(f"{'=' * 80}")
        print(f"Number of samples: {len(split_data)}")
        print(f"Columns: {split_data.column_names}")

        # Inspect first few examples
        print(f"\nFirst 3 examples:")
        for i in range(min(3, len(split_data))):
            print(f"\n--- Example {i} ---")
            example = split_data[i]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 200:
                    print(f"  {key}: {value[:200]}... (truncated)")
                else:
                    print(f"  {key}: {value}")

    # 3. Detailed column analysis
    print("\n" + "=" * 80)
    print("2. DETAILED COLUMN ANALYSIS")
    print("=" * 80)

    for split_name, split_data in dataset.items():
        print(f"\n{'=' * 80}")
        print(f"SPLIT: {split_name}")
        print(f"{'=' * 80}")

        # Get column info
        features = split_data.features
        print(f"\nColumn types:")
        for col_name, col_type in features.items():
            print(f"  {col_name}: {col_type}")

        # Sample analysis
        if len(split_data) > 0:
            sample = split_data[0]
            print(f"\nSample record structure:")
            for key, value in sample.items():
                value_type = type(value).__name__
                if isinstance(value, str):
                    print(f"  {key}: {value_type} (length: {len(value)})")
                elif isinstance(value, list):
                    print(f"  {key}: {value_type} (length: {len(value)})")
                    if len(value) > 0:
                        print(f"    First element type: {type(value[0])}")
                        if isinstance(value[0], str):
                            print(f"    First element: {value[0][:100]}...")
                else:
                    print(f"  {key}: {value_type} = {value}")

    # 4. Identify key fields
    print("\n" + "=" * 80)
    print("3. KEY FIELD IDENTIFICATION")
    print("=" * 80)

    # Check for question/query fields
    print("\nPossible question/query fields:")
    for split_name, split_data in dataset.items():
        if len(split_data) > 0:
            sample = split_data[0]
            for key in ['question', 'query', 'questions', 'queries', 'text', 'input']:
                if key in sample:
                    print(f"  {split_name}.{key}: found")

    # Check for context/document fields
    print("\nPossible context/document fields:")
    for split_name, split_data in dataset.items():
        if len(split_data) > 0:
            sample = split_data[0]
            for key in ['context', 'contexts', 'document', 'documents', 'passage', 'passages', 'text', 'content']:
                if key in sample:
                    print(f"  {split_name}.{key}: found")

    # Check for language fields
    print("\nPossible language fields:")
    for split_name, split_data in dataset.items():
        if len(split_data) > 0:
            sample = split_data[0]
            for key in ['language', 'lang', 'locale', 'locale_name']:
                if key in sample:
                    print(f"  {split_name}.{key}: found")
                    # Get unique languages if possible
                    try:
                        unique_langs = set(split_data[key])
                        print(f"    Unique values: {unique_langs}")
                    except:
                        pass

    # Check for IDs
    print("\nPossible ID fields:")
    for split_name, split_data in dataset.items():
        if len(split_data) > 0:
            sample = split_data[0]
            for key in ['id', 'query_id', 'doc_id', 'document_id', 'qid', 'did']:
                if key in sample:
                    print(f"  {split_name}.{key}: found")

    # 5. Save schema to file
    schema = {
        "splits": list(dataset.keys()),
        "split_sizes": {name: len(data) for name, data in dataset.items()},
        "columns": {name: data.column_names for name, data in dataset.items()},
        "features": {name: {col: str(ft) for col, ft in data.features.items()} for name, data in dataset.items()},
    }

    with open("dataset_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print("\n" + "=" * 80)
    print("Schema saved to dataset_schema.json")
    print("=" * 80)

    return dataset

if __name__ == "__main__":
    inspect_dataset()
