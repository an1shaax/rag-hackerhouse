# RAG Pipeline Latency Report

**Generated:** 2026-08-20T15:42:58.610787

**Total Queries:** 50

## Summary

- **Successful:** 50
- **Failed:** 0
- **Refused:** 44

## Latency Percentiles

| Metric | Latency (ms) |
|--------|-------------|
| P50 | 45.48 |
| P70 | 46.56 |
| P90 | 59.04 |
| P95 | 66.25 |
| P99 | 75.81 |
| P100 (Max) | 78.10 |
| Mean | 47.59 |
| Min | 37.37 |

## Component Latencies (Average)

| Component | Latency (ms) |
|-----------|-------------|
| embedding_ms | 15.08 |
| retrieval_ms | 0.24 |
| reranking_ms | 32.22 |
| generation_ms | 0.01 |
| grounding_ms | 0.04 |

## Target Analysis

✅ **P50 latency (45.48ms) is UNDER 200ms target**

