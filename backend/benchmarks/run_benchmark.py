#!/usr/bin/env python3
"""
Benchmarking script for measuring RAG pipeline latency.

Measures:
- Query embedding latency
- Retrieval latency
- Reranking latency
- Generation latency
- Grounding latency
- Total RAG latency
- End-to-end latency

Calculates P50, P70, P90, P95, P99, P100 percentiles.
"""
import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import time
import uuid
import csv
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import structlog

from app.config import get_settings
from app.models.schemas import (
    QueryRequest, BenchmarkResult, BenchmarkReport, Language
)
from app.services.embeddings import get_embedding_service
from app.services.retrieval import get_retrieval_service
from app.services.reranking import get_reranker_service
from app.services.generation import get_generation_service
from app.services.guardrails import get_guardrails_service

logger = structlog.get_logger()


class RAGBenchmark:
    """Benchmark RAG pipeline performance"""

    def __init__(self):
        self.settings = get_settings()
        self.embedding_service = get_embedding_service()
        self.retrieval_service = get_retrieval_service()
        self.reranker_service = get_reranker_service()
        self.generation_service = get_generation_service()
        self.guardrails_service = get_guardrails_service()

    def load_test_queries(self, num_queries: int = 100) -> List[Dict[str, str]]:
        """Load test queries - use sample queries to avoid dataset loading issues"""
        logger.info("loading_test_queries", num_queries=num_queries)

        # Use predefined sample queries covering various topics
        sample_queries = [
            {"query": "What is the capital of India?", "query_id": "q1", "language": "en"},
            {"query": "Who is the prime minister of India?", "query_id": "q2", "language": "en"},
            {"query": "What is the population of India?", "query_id": "q3", "language": "en"},
            {"query": "What are the major religions in India?", "query_id": "q4", "language": "en"},
            {"query": "What is the currency of India?", "query_id": "q5", "language": "en"},
            {"query": "What is the national animal of India?", "query_id": "q6", "language": "en"},
            {"query": "What are the official languages of India?", "query_id": "q7", "language": "en"},
            {"query": "What is the climate of India?", "query_id": "q8", "language": "en"},
            {"query": "What is the geography of India?", "query_id": "q9", "language": "en"},
            {"query": "What is the history of India?", "query_id": "q10", "language": "en"},
            {"query": "What is the economy of India?", "query_id": "q11", "language": "en"},
            {"query": "What is the culture of India?", "query_id": "q12", "language": "en"},
            {"query": "What are the major cities in India?", "query_id": "q13", "language": "en"},
            {"query": "What is the education system in India?", "query_id": "q14", "language": "en"},
            {"query": "What is the healthcare system in India?", "query_id": "q15", "language": "en"},
        ]

        # Repeat and vary queries to reach num_queries
        queries = []
        for i in range(num_queries):
            base_query = sample_queries[i % len(sample_queries)]
            queries.append({
                "query": base_query["query"],
                "query_id": f"{base_query['query_id']}_{i}",
                "language": base_query["language"]
            })

        logger.info("test_queries_loaded", count=len(queries))
        return queries

    def run_single_query(self, query_data: Dict[str, str]) -> BenchmarkResult:
        """Run a single query through the pipeline and measure latencies"""
        query_id = query_data["query_id"]
        query = query_data["query"]
        language = query_data.get("language", "en")

        latencies = {}
        total_start = time.time()

        try:
            # 1. Query embedding
            embed_start = time.time()
            query_embedding = self.embedding_service.embed_single(query)
            latencies["embedding"] = (time.time() - embed_start) * 1000

            # 2. Retrieval
            retrieval_start = time.time()
            # Use top_k=10 for benchmarking (more realistic for production)
            # Map language codes: "en" -> "eng" (index uses "eng")
            lang_map = {"en": "eng", "hi": "hin", "bn": "ben", "ta": "tam", "te": "tel"}
            search_lang = lang_map.get(language, language)
            chunks, _ = self.retrieval_service.search(query_embedding, top_k=10, language=search_lang)
            latencies["retrieval"] = (time.time() - retrieval_start) * 1000

            # 3. Reranking
            rerank_start = time.time()
            if chunks:
                reranked_chunks, _ = self.reranker_service.rerank(query, chunks)
            else:
                reranked_chunks = []
            latencies["reranking"] = (time.time() - rerank_start) * 1000

            # 4. Generation
            generation_start = time.time()
            if reranked_chunks:
                answer, _ = self.generation_service.generate(query, reranked_chunks, language)
            else:
                answer = None
            latencies["generation"] = (time.time() - generation_start) * 1000

            # 5. Grounding
            grounding_start = time.time()
            if answer:
                is_grounded, _ = self.guardrails_service.verify_grounding(answer, reranked_chunks, query)
            else:
                is_grounded = False
            latencies["grounding"] = (time.time() - grounding_start) * 1000

            # Total RAG latency (excluding STT)
            total_rag = (time.time() - total_start) * 1000

            return BenchmarkResult(
                query_id=query_id,
                query=query[:100],
                embedding_latency_ms=latencies["embedding"],
                retrieval_latency_ms=latencies["retrieval"],
                reranking_latency_ms=latencies["reranking"],
                generation_latency_ms=latencies["generation"],
                grounding_latency_ms=latencies["grounding"],
                total_rag_latency_ms=total_rag,
                total_latency_ms=total_rag,
                success=True,
                refused=not is_grounded if answer else True,
                answer_length=len(answer.answer) if answer else 0,
                num_citations=len(answer.citations) if answer else 0
            )

        except Exception as e:
            logger.error("benchmark_query_error", query_id=query_id, error=str(e))
            return BenchmarkResult(
                query_id=query_id,
                query=query[:100],
                embedding_latency_ms=latencies.get("embedding", 0),
                retrieval_latency_ms=latencies.get("retrieval", 0),
                reranking_latency_ms=latencies.get("reranking", 0),
                generation_latency_ms=latencies.get("generation", 0),
                grounding_latency_ms=latencies.get("grounding", 0),
                total_rag_latency_ms=(time.time() - total_start) * 1000,
                total_latency_ms=(time.time() - total_start) * 1000,
                success=False,
                refused=True,
                answer_length=0,
                num_citations=0
            )

    def calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentile statistics"""
        if not values:
            return {}

        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p70": float(np.percentile(arr, 70)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "p100": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def run_benchmark(self, num_queries: int = 100, warmup_queries: int = 5) -> BenchmarkReport:
        """Run full benchmark suite with warmup"""
        logger.info("starting_benchmark", num_queries=num_queries, warmup=warmup_queries)

        # Load test queries
        queries = self.load_test_queries(num_queries + warmup_queries)

        # Warmup phase - run a few queries to load models
        logger.info("running_warmup", count=warmup_queries)
        for query_data in tqdm(queries[:warmup_queries], desc="Warmup"):
            self.run_single_query(query_data)

        # Actual benchmark
        logger.info("running_benchmark", count=num_queries)
        results = []
        for query_data in tqdm(queries[warmup_queries:], desc="Running benchmark"):
            result = self.run_single_query(query_data)
            results.append(result)

        # Calculate statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        refused = [r for r in results if r.refused]

        total_latencies = [r.total_rag_latency_ms for r in successful]
        percentiles = self.calculate_percentiles(total_latencies)

        # Component latencies
        component_latencies = {
            "embedding_ms": float(np.mean([r.embedding_latency_ms for r in successful])) if successful else 0,
            "retrieval_ms": float(np.mean([r.retrieval_latency_ms for r in successful])) if successful else 0,
            "reranking_ms": float(np.mean([r.reranking_latency_ms for r in successful])) if successful else 0,
            "generation_ms": float(np.mean([r.generation_latency_ms for r in successful])) if successful else 0,
            "grounding_ms": float(np.mean([r.grounding_latency_ms for r in successful])) if successful else 0,
        }

        report = BenchmarkReport(
            total_queries=len(results),
            successful_queries=len(successful),
            failed_queries=len(failed),
            refused_queries=len(refused),
            p50_latency_ms=percentiles.get("p50", 0),
            p70_latency_ms=percentiles.get("p70", 0),
            p90_latency_ms=percentiles.get("p90", 0),
            p95_latency_ms=percentiles.get("p95", 0),
            p99_latency_ms=percentiles.get("p99", 0),
            p100_latency_ms=percentiles.get("p100", 0),
            mean_latency_ms=percentiles.get("mean", 0),
            min_latency_ms=percentiles.get("min", 0),
            max_latency_ms=percentiles.get("max", 0),
            avg_component_latencies=component_latencies,
            results=results
        )

        return report


def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries to benchmark")
    parser.add_argument("--warmup-queries", type=int, default=5, help="Number of warmup queries to run before benchmark")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")

    args = parser.parse_args()

    # Initialize services
    logger.info("initializing_services")
    embedding_service = get_embedding_service()
    embedding_service.load_model()

    retrieval_service = get_retrieval_service()
    index_loaded = retrieval_service.load_index()

    if not index_loaded:
        logger.warning("index_not_loaded_using_mock_mode")

    # Run benchmark
    benchmark = RAGBenchmark()
    report = benchmark.run_benchmark(num_queries=args.num_queries, warmup_queries=args.warmup_queries)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = output_dir / "latency.json"
    with open(json_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)

    # Save CSV
    csv_path = output_dir / "latency.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "query_id", "query", "embedding_latency_ms", "retrieval_latency_ms",
            "reranking_latency_ms", "generation_latency_ms", "grounding_latency_ms",
            "total_rag_latency_ms", "total_latency_ms", "success", "refused"
        ])
        writer.writeheader()
        for result in report.results:
            writer.writerow({
                "query_id": result.query_id,
                "query": result.query,
                "embedding_latency_ms": result.embedding_latency_ms,
                "retrieval_latency_ms": result.retrieval_latency_ms,
                "reranking_latency_ms": result.reranking_latency_ms,
                "generation_latency_ms": result.generation_latency_ms,
                "grounding_latency_ms": result.grounding_latency_ms,
                "total_rag_latency_ms": result.total_rag_latency_ms,
                "total_latency_ms": result.total_latency_ms,
                "success": result.success,
                "refused": result.refused,
            })

    # Generate markdown report
    md_path = output_dir / "latency_report.md"
    with open(md_path, "w") as f:
        f.write("# RAG Pipeline Latency Report\n\n")
        f.write(f"**Generated:** {datetime.utcnow().isoformat()}\n\n")
        f.write(f"**Total Queries:** {report.total_queries}\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Successful:** {report.successful_queries}\n")
        f.write(f"- **Failed:** {report.failed_queries}\n")
        f.write(f"- **Refused:** {report.refused_queries}\n\n")

        f.write("## Latency Percentiles\n\n")
        f.write(f"| Metric | Latency (ms) |\n")
        f.write(f"|--------|-------------|\n")
        f.write(f"| P50 | {report.p50_latency_ms:.2f} |\n")
        f.write(f"| P70 | {report.p70_latency_ms:.2f} |\n")
        f.write(f"| P90 | {report.p90_latency_ms:.2f} |\n")
        f.write(f"| P95 | {report.p95_latency_ms:.2f} |\n")
        f.write(f"| P99 | {report.p99_latency_ms:.2f} |\n")
        f.write(f"| P100 (Max) | {report.p100_latency_ms:.2f} |\n")
        f.write(f"| Mean | {report.mean_latency_ms:.2f} |\n")
        f.write(f"| Min | {report.min_latency_ms:.2f} |\n\n")

        f.write("## Component Latencies (Average)\n\n")
        f.write(f"| Component | Latency (ms) |\n")
        f.write(f"|-----------|-------------|\n")
        for component, latency in report.avg_component_latencies.items():
            f.write(f"| {component} | {latency:.2f} |\n")
        f.write("\n")

        f.write("## Target Analysis\n\n")
        if report.p50_latency_ms < 200:
            f.write(f"✅ **P50 latency ({report.p50_latency_ms:.2f}ms) is UNDER 200ms target**\n\n")
        else:
            f.write(f"⚠️ **P50 latency ({report.p50_latency_ms:.2f}ms) EXCEEDS 200ms target**\n\n")

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"\nTotal queries: {report.total_queries}")
    print(f"Successful: {report.successful_queries}")
    print(f"Failed: {report.failed_queries}")
    print(f"Refused: {report.refused_queries}")
    print("\nLatency Percentiles:")
    print(f"  P50:  {report.p50_latency_ms:.2f}ms")
    print(f"  P70:  {report.p70_latency_ms:.2f}ms")
    print(f"  P90:  {report.p90_latency_ms:.2f}ms")
    print(f"  P99:  {report.p99_latency_ms:.2f}ms")
    print(f"  P100: {report.p100_latency_ms:.2f}ms")
    print(f"  Mean: {report.mean_latency_ms:.2f}ms")

    print(f"\n{'✅' if report.p50_latency_ms < 200 else '⚠️'} Target: <200ms")
    print(f"\nReports saved to {output_dir}/")


if __name__ == "__main__":
    main()
