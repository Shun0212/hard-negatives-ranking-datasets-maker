"""Sentence-Transformers encoder with FAISS index."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)


class SentenceTransformerEncoder(BaseEncoder):
    """Encode queries/documents with SentenceTransformer and retrieve via FAISS."""

    def __init__(
        self,
        model_name: str,
        index_dir: str = "./faiss_index",
        encode_batch_size: int = 32,
        device: str | None = None,
        faiss_index_type: str = "flat",
        faiss_metric: str = "cosine",
        faiss_nlist: int = 100,
        faiss_nprobe: int = 10,
        faiss_m_pq: int = 8,
        faiss_hnsw_m: int = 32,
        faiss_ef_search: int = 64,
        faiss_use_gpu: bool = False,
    ):
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index_dir = index_dir
        self.encode_batch_size = encode_batch_size
        self.faiss_index_type = faiss_index_type
        self.faiss_metric = faiss_metric
        self.faiss_nlist = faiss_nlist
        self.faiss_nprobe = faiss_nprobe
        self.faiss_m_pq = faiss_m_pq
        self.faiss_hnsw_m = faiss_hnsw_m
        self.faiss_ef_search = faiss_ef_search
        self.faiss_use_gpu = faiss_use_gpu

        self.index: Optional[faiss.Index] = None
        self.doc_id_mapping: Dict[int, str] = {}
        self.total_docs: int = 0

    def _get_faiss_metric(self) -> int:
        """Map string metric to FAISS metric constant."""
        if self.faiss_metric in ("cosine", "ip"):
            return faiss.METRIC_INNER_PRODUCT
        elif self.faiss_metric == "l2":
            return faiss.METRIC_L2
        else:
            raise ValueError(f"Unknown metric: {self.faiss_metric}")

    def _build_index(self, dim: int, num_docs: int) -> faiss.Index:
        """Build the appropriate FAISS index based on configuration."""
        metric = self._get_faiss_metric()

        if self.faiss_index_type == "flat":
            index = faiss.IndexFlat(dim, metric)

        elif self.faiss_index_type == "ivf":
            nlist = min(self.faiss_nlist, max(num_docs // 10, 1))
            quantizer = faiss.IndexFlat(dim, metric)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)

        elif self.faiss_index_type == "ivfpq":
            nlist = min(self.faiss_nlist, max(num_docs // 10, 1))
            quantizer = faiss.IndexFlat(dim, metric)
            index = faiss.IndexIVFPQ(
                quantizer, dim, nlist, self.faiss_m_pq, 8, metric
            )

        elif self.faiss_index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.faiss_hnsw_m, metric)
            index.hnsw.efSearch = self.faiss_ef_search

        else:
            raise ValueError(
                f"Unknown FAISS index type: {self.faiss_index_type}"
            )

        return index

    def encode_and_index_documents(
        self,
        documents: List[str],
        document_ids: List[str],
        index_name: str = "documents",
        batch_size: int = 500,
    ) -> None:
        """Encode documents with SentenceTransformer and build a FAISS index."""
        logger.info(
            f"Encoding {len(documents)} documents with SentenceTransformer..."
        )

        os.makedirs(self.index_dir, exist_ok=True)

        # Reset state
        self.doc_id_mapping = {}
        self.total_docs = len(documents)
        for i, did in enumerate(document_ids):
            self.doc_id_mapping[i] = did

        # Encode all documents
        embeddings = self.model.encode(
            documents,
            batch_size=self.encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=(self.faiss_metric == "cosine"),
        )
        embeddings = embeddings.astype(np.float32)

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = self._build_index(dim, len(documents))

        # Train index if needed (IVF variants require training)
        if not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)

        # Add vectors
        self.index.add(embeddings)

        # Optionally move to GPU
        if self.faiss_use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        # Save index to disk
        index_path = os.path.join(self.index_dir, f"{index_name}.faiss")
        if not self.faiss_use_gpu:
            faiss.write_index(self.index, index_path)
            logger.info(f"FAISS index saved to {index_path}")

        logger.info(
            f"Document indexing complete. "
            f"Total: {self.total_docs} documents, dim={dim}, "
            f"index_type={self.faiss_index_type}"
        )

    def retrieve(
        self,
        queries: List[str],
        query_ids: List[str],
        top_k: int = 200,
        batch_size: int = 500,
        search_batch_size: int = 25000,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve top-K documents for each query using FAISS search."""
        if self.index is None:
            raise RuntimeError(
                "Index not built. Call encode_and_index_documents first."
            )

        logger.info(f"Retrieving top-{top_k} for {len(queries)} queries...")

        # Set IVF nprobe if applicable
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.faiss_nprobe

        all_results: Dict[str, List[Tuple[str, float]]] = {}

        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            batch_queries = queries[start:end]
            batch_qids = query_ids[start:end]

            logger.info(
                f"  Retrieving for queries [{start}:{end}] / {len(queries)}"
            )

            query_embeddings = self.model.encode(
                batch_queries,
                batch_size=self.encode_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=(self.faiss_metric == "cosine"),
            )
            query_embeddings = query_embeddings.astype(np.float32)

            # FAISS search
            scores, indices = self.index.search(query_embeddings, top_k)

            for i, qid in enumerate(batch_qids):
                results = []
                for j in range(top_k):
                    idx = int(indices[i][j])
                    if idx == -1:  # FAISS returns -1 for missing results
                        continue
                    score = float(scores[i][j])
                    doc_id = self.doc_id_mapping[idx]
                    results.append((doc_id, score))
                all_results[qid] = results

        logger.info("Retrieval complete.")
        return all_results
