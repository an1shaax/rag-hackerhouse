#!/usr/bin/env python3
"""
Data ingestion pipeline for MSMARCO-XI dataset

This script:
1. Downloads/loads MSMARCO-XI dataset from HuggingFace
2. Normalizes documents
3. Chunks documents using multiple strategies
4. Generates embeddings
5. Builds FAISS index
6. Saves index and metadata

Usage:
    python build_index.py --strategy semantic --limit 10000 --split train
"""
import argparse
import json
import pickle
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import time
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
import structlog

from ingestion.chunkers import get_chunker, ChunkResult
from app.services.embeddings import get_embedding_service

logger = structlog.get_logger()


class DataIngestionPipeline:
    """Pipeline for ingesting MSMARCO-XI dataset"""

    def __init__(
        self,
        output_dir: str = "indexes",
        chunk_strategy: str = "semantic",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        limit: int = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.limit = limit

        self.chunker = get_chunker(chunk_strategy, chunk_size, chunk_overlap)
        self.embedding_service = get_embedding_service()

        self.chunks: List[ChunkResult] = []
        self.chunk_metadata: Dict[int, Dict[str, Any]] = {}

    def load_dataset(self, split: str = "train"):
        """
        Load dataset - tries synthetic data first, then falls back to HuggingFace

        Args:
            split: Dataset split to load (ignored for synthetic)
        """
        logger.info("loading_dataset", split=split)

        # Try to load synthetic data first
        # Check multiple possible paths
        synthetic_paths = [
            Path("../data/synthetic_msmarco_xi.json"),  # from backend/ingestion/
            Path("data/synthetic_msmarco_xi.json"),     # from backend/
            Path("../../data/synthetic_msmarco_xi.json"),  # from other locations
        ]
        synthetic_path = None
        for p in synthetic_paths:
            if p.exists():
                synthetic_path = p
                break
        if synthetic_path and synthetic_path.exists():
            logger.info("loading_synthetic_dataset", path=str(synthetic_path))
            with open(synthetic_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            if self.limit:
                records = records[:self.limit]

            # Convert to dataset-like object with select method
            class SyntheticDataset:
                def __init__(self, data):
                    self.data = data

                def __len__(self):
                    return len(self.data)

                def __iter__(self):
                    return iter(self.data)

                def select(self, indices):
                    return SyntheticDataset([self.data[i] for i in indices])

            logger.info("synthetic_dataset_loaded", num_rows=len(records))
            return SyntheticDataset(records)

        # Fall back to HuggingFace
        try:
            dataset = load_dataset("ai4bharat/MSMARCO-XI", split=split)
            logger.info("dataset_loaded", num_rows=len(dataset))

            if self.limit:
                dataset = dataset.select(range(min(self.limit, len(dataset))))
                logger.info("dataset_limited", limit=self.limit)

            return dataset

        except Exception as e:
            logger.error("dataset_load_error", error=str(e))
            raise

    def process_record(self, record: Dict[str, Any]) -> List[ChunkResult]:
        """
        Process a single record from the dataset

        MSMARCO-XI schema:
        - query: translated search query
        - Answer: translated answer
        - query_id: unique query identifier
        - query_type: category of query
        - passages.is_selected: list of binary selection indicators
        - passages.English_passages: list of original English passages
        - passages.Translated_passages: list of translated passages
        - source_lang: source language code
        - target_lang: target language code

        Args:
            record: Dataset record

        Returns:
            List of chunks
        """
        document_id = str(record.get("query_id", uuid.uuid4()))

        # Extract language
        target_lang = record.get("target_lang", "en")
        # Map language code (e.g., "hi" from "hin_Deva")
        lang_code = target_lang.split("_")[0] if "_" in target_lang else target_lang

        metadata = {
            "query_id": document_id,
            "query_type": record.get("query_type", "unknown"),
            "language": lang_code,
            "source": "MSMARCO-XI",
            "source_lang": record.get("source_lang", "eng_Latn"),
            "target_lang": target_lang,
        }

        chunks = []

        # Skip chunking the query - we don't want queries in the retrieval index
        # as they cause exact matches when users ask the same questions

        # Chunk the answer in the target language
        answer = record.get("Answer", "")
        if answer:
            answer_chunks = self.chunker.chunk(
                text=answer,
                document_id=f"{document_id}_answer",
                metadata={**metadata, "content_type": "answer"}
            )
            chunks.extend(answer_chunks)

        # Also chunk the English answer if available (for better English query coverage)
        eng_answer = record.get("Eng_Answer", "")
        if eng_answer and eng_answer != answer:
            eng_metadata = {**metadata, "language": "eng", "content_type": "answer"}
            answer_chunks = self.chunker.chunk(
                text=eng_answer,
                document_id=f"{document_id}_answer_eng",
                metadata=eng_metadata
            )
            chunks.extend(answer_chunks)

        # Chunk passages
        passages = record.get("passages", {})
        translated_passages = passages.get("Translated_passages", [])
        english_passages = passages.get("English_passages", [])
        is_selected = passages.get("is_selected", [])

        # Chunk translated passages
        for i, passage in enumerate(translated_passages):
            if passage and (not is_selected or (i < len(is_selected) and is_selected[i])):
                passage_chunks = self.chunker.chunk(
                    text=passage,
                    document_id=f"{document_id}_passage_{i}",
                    metadata={
                        **metadata,
                        "content_type": "passage",
                        "passage_index": i,
                        "is_selected": is_selected[i] if i < len(is_selected) else True
                    }
                )
                chunks.extend(passage_chunks)

        # Also chunk English passages for better English query coverage
        for i, passage in enumerate(english_passages):
            if passage and (not is_selected or (i < len(is_selected) and is_selected[i])):
                eng_metadata = {
                    **metadata,
                    "language": "eng",
                    "content_type": "passage",
                    "passage_index": i,
                    "is_selected": is_selected[i] if i < len(is_selected) else True
                }
                passage_chunks = self.chunker.chunk(
                    text=passage,
                    document_id=f"{document_id}_passage_eng_{i}",
                    metadata=eng_metadata
                )
                chunks.extend(passage_chunks)

        return chunks

    def ingest_dataset(self, split: str = "train"):
        """
        Ingest complete dataset

        Args:
            split: Dataset split to ingest
        """
        start_time = time.time()

        # Load dataset
        dataset = self.load_dataset(split)

        logger.info(
            "starting_ingestion",
            split=split,
            num_records=len(dataset),
            chunk_strategy=self.chunk_strategy
        )

        # Process records
        all_chunks = []

        for record in tqdm(dataset, desc="Processing records"):
            chunks = self.process_record(record)
            all_chunks.extend(chunks)

        # Deduplicate chunks by (text_hash, language) to preserve multilingual content
        # but remove exact duplicates within the same language
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            # Use text hash + language as key to allow same text in different languages
            text_hash = hash(chunk.text.strip().lower())
            lang = getattr(chunk, 'language', 'unknown')
            key = (text_hash, lang)
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)

        logger.info(
            "deduplication_complete",
            original_chunks=len(all_chunks),
            unique_chunks=len(unique_chunks),
            removed=len(all_chunks) - len(unique_chunks)
        )

        self.chunks = unique_chunks

        ingestion_time = time.time() - start_time

        logger.info(
            "ingestion_complete",
            num_chunks=len(self.chunks),
            ingestion_time_seconds=ingestion_time
        )

        return self.chunks

    def build_index(self):
        """
        Build FAISS index from chunks

        Returns:
            FAISS index
        """
        if not self.chunks:
            raise ValueError("No chunks to index. Run ingest_dataset first.")

        logger.info("building_index", num_chunks=len(self.chunks))

        start_time = time.time()

        # Generate embeddings
        texts = [chunk.text for chunk in self.chunks]

        logger.info("generating_embeddings", num_texts=len(texts))
        embeddings = self.embedding_service.embed(texts, show_progress=True)

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity (normalized vectors)

        # Add vectors to index
        index.add(embeddings.astype('float32'))

        # Build metadata mapping
        self.chunk_metadata = {}
        for i, chunk in enumerate(self.chunks):
            self.chunk_metadata[i] = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "language": chunk.language,
                "source": chunk.source,
                "chunking_strategy": chunk.chunking_strategy,
                "position": chunk.position,
                "metadata": chunk.metadata
            }

        build_time = time.time() - start_time

        logger.info(
            "index_built",
            num_vectors=index.ntotal,
            dimension=dimension,
            build_time_seconds=build_time
        )

        return index

    def save_index(self, index: faiss.Index):
        """
        Save FAISS index and metadata to disk

        Args:
            index: FAISS index to save
        """
        logger.info("saving_index", output_dir=str(self.output_dir))

        # Save FAISS index
        index_path = self.output_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))

        # Save metadata
        metadata_path = self.output_dir / "chunk_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.chunk_metadata, f)

        # Save ID mapping
        id_mapping_path = self.output_dir / "id_mapping.json"
        id_mapping = {str(i): chunk.chunk_id for i, chunk in enumerate(self.chunks)}
        with open(id_mapping_path, "w") as f:
            json.dump(id_mapping, f)

        # Save ingestion stats
        stats = {
            "num_chunks": len(self.chunks),
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_size": index.ntotal,
            "embedding_dim": index.d,
        }
        stats_path = self.output_dir / "ingestion_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            "index_saved",
            index_path=str(index_path),
            num_vectors=index.ntotal
        )


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Build FAISS index from MSMARCO-XI dataset")

    parser.add_argument(
        "--strategy",
        type=str,
        default="semantic",
        choices=["fixed", "fixed_overlap", "sentence", "semantic", "metadata_aware"],
        help="Chunking strategy to use"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in characters"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (for testing)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="indexes",
        help="Output directory for index files"
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = DataIngestionPipeline(
        output_dir=args.output,
        chunk_strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        limit=args.limit
    )

    # Ingest dataset
    pipeline.ingest_dataset(split=args.split)

    # Build index
    index = pipeline.build_index()

    # Save index
    pipeline.save_index(index)

    print(f"\n✅ Index built successfully!")
    print(f"   Strategy: {args.strategy}")
    print(f"   Chunks: {len(pipeline.chunks)}")
    print(f"   Output: {args.output}/")


if __name__ == "__main__":
    main()
