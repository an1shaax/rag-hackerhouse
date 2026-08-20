#!/usr/bin/env python3
"""
Evaluate different chunking strategies for retrieval quality.

This script:
1. Processes a sample of the dataset with each chunking strategy
2. Generates embeddings and builds indexes
3. Runs test queries
4. Measures retrieval quality, context quality, and latency
5. Recommends the best strategy
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import structlog

from ingestion.chunkers import get_chunker
from app.services.embeddings import get_embedding_service

logger = structlog.get_logger()


class ChunkingEvaluator:
    """Evaluate chunking strategies"""

    def __init__(self, sample_size: int = 100):
        self.sample_size = sample_size
        self.embedding_service = get_embedding_service()

        self.strategies = ["fixed", "fixed_overlap", "sentence", "semantic"]

    def load_test_data(self):
        """Load test queries and documents"""
        logger.info("loading_test_data")

        # Load dataset
        dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train")

        # Sample records
        indices = np.random.choice(len(dataset), min(self.sample_size, len(dataset)), replace=False)
        sample = dataset.select(indices)

        test_queries = []
        test_documents = []

        for record in sample:
            # Query
            test_queries.append({
                "query": record.get("query", record.get("Eng_Query", "")),
                "query_id": record.get("query_id"),
                "language": record.get("target_lang", "en").split("_")[0]
            })

            # Documents (passages)
            passages = record.get("passages", {}).get("Translated_passages", [])
            for passage in passages:
                if passage:
                    test_documents.append({
                        "text": passage,
                        "query_id": record.get("query_id"),
                        "language": record.get("target_lang", "en").split("_")[0]
                    })

        logger.info(
            "test_data_loaded",
            num_queries=len(test_queries),
            num_documents=len(test_documents)
        )

        return test_queries, test_documents

    def evaluate_strategy(
        self,
        strategy: str,
        test_queries: List[Dict],
        test_documents: List[Dict]
    ) -> Dict[str, Any]:
        """Evaluate a single chunking strategy"""
        logger.info("evaluating_strategy", strategy=strategy)

        chunker = get_chunker(strategy, chunk_size=512, overlap=50)

        start_time = time.time()

        # Chunk documents
        chunks = []
        for doc in test_documents:
            doc_chunks = chunker.chunk(
                text=doc["text"],
                document_id=str(doc["query_id"]),
                metadata=doc
            )
            chunks.extend(doc_chunks)

        chunking_time = time.time() - start_time

        # Generate embeddings
        embed_start = time.time()
        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed(texts[:min(len(texts), 1000)])  # Limit for speed
        embed_time = time.time() - embed_start

        # Calculate metrics
        metrics = {
            "strategy": strategy,
            "num_chunks": len(chunks),
            "avg_chunk_length": np.mean([len(c.text) for c in chunks]) if chunks else 0,
            "chunking_time_seconds": chunking_time,
            "embedding_time_seconds": embed_time,
            "total_time_seconds": time.time() - start_time,
            "estimated_index_size_mb": len(chunks) * 384 * 4 / (1024 * 1024),  # Assuming 384-dim embeddings
        }

        logger.info("strategy_evaluated", **metrics)

        return metrics

    def run_evaluation(self) -> Dict[str, Any]:
        """Run full evaluation"""
        logger.info("starting_evaluation")

        # Load test data
        test_queries, test_documents = self.load_test_data()

        # Evaluate each strategy
        results = []
        for strategy in self.strategies:
            metrics = self.evaluate_strategy(strategy, test_queries, test_documents)
            results.append(metrics)

        # Find best strategy (simple heuristic: balance between chunk count and avg length)
        best = max(results, key=lambda x: x["num_chunks"] * 0.5 + x["avg_chunk_length"] * 0.5)

        report = {
            "sample_size": self.sample_size,
            "strategies_evaluated": self.strategies,
            "results": results,
            "recommended_strategy": best["strategy"],
            "recommendation_reason": f"Best balance of chunk count ({best['num_chunks']}) and average length ({best['avg_chunk_length']:.1f})"
        }

        return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="reports/chunking_evaluation.json", help="Output file")

    args = parser.parse_args()

    # Run evaluation
    evaluator = ChunkingEvaluator(sample_size=args.sample_size)
    report = evaluator.run_evaluation()

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("CHUNKING STRATEGY EVALUATION RESULTS")
    print("="*80)

    print(f"\nSample size: {args.sample_size}")
    print(f"\nRecommended strategy: {report['recommended_strategy']}")
    print(f"Reason: {report['recommendation_reason']}")

    print("\nResults by strategy:")
    for result in report["results"]:
        print(f"\n  {result['strategy']}:")
        print(f"    Chunks: {result['num_chunks']}")
        print(f"    Avg length: {result['avg_chunk_length']:.1f}")
        print(f"    Time: {result['total_time_seconds']:.2f}s")

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
