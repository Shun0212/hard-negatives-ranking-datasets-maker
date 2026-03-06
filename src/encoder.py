"""ColBERT encoder and retriever using PyLate + fast-plaid."""

import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from fast_plaid import search as fp_search
from pylate import models

logger = logging.getLogger(__name__)


def _to_tensors(embeddings: list) -> List[torch.Tensor]:
    """Convert model encode output to a list of torch.Tensor."""
    result = []
    for e in embeddings:
        if isinstance(e, torch.Tensor):
            result.append(e.float())
        elif isinstance(e, np.ndarray):
            result.append(torch.from_numpy(e).float())
        else:
            result.append(torch.tensor(np.array(e), dtype=torch.float32))
    return result


def _stack_queries(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Stack variable-length query embeddings into a padded batch tensor.

    ColBERT queries are typically padded to the same length, but we handle
    the variable-length case as a safeguard.
    """
    max_len = max(t.shape[0] for t in tensors)
    dim = tensors[0].shape[1]
    batch = torch.zeros(len(tensors), max_len, dim)
    for i, t in enumerate(tensors):
        batch[i, : t.shape[0]] = t
    return batch


class ColBERTEncoder:
    """Encode queries/documents with ColBERT and retrieve via fast-plaid."""

    def __init__(
        self,
        model_name: str,
        index_dir: str = "./plaid_index",
        encode_batch_size: int = 32,
        device: str | None = None,
    ):
        logger.info(f"Loading ColBERT model: {model_name}")
        self.model = models.ColBERT(model_name_or_path=model_name)
        self.index_dir = index_dir
        self.encode_batch_size = encode_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.fp_index: fp_search.FastPlaid | None = None
        self.doc_id_mapping: Dict[int, str] = {}
        self.total_docs: int = 0

    def encode_and_index_documents(
        self,
        documents: List[str],
        document_ids: List[str],
        index_name: str = "documents",
        batch_size: int = 500,
    ) -> None:
        """Encode documents with ColBERT and build a fast-plaid index.

        Encodes in batches for GPU memory efficiency, then creates the
        fast-plaid index with ALL documents at once so that K-means
        centroids are computed from the full corpus.
        """
        logger.info(f"Encoding {len(documents)} documents...")

        os.makedirs(self.index_dir, exist_ok=True)
        index_path = os.path.join(self.index_dir, index_name)

        # Reset state
        self.doc_id_mapping = {}
        self.total_docs = len(documents)

        # Build doc_id mapping (fast-plaid uses sequential int indices)
        for i, did in enumerate(document_ids):
            self.doc_id_mapping[i] = did

        # Step 1: Encode all documents in batches, accumulate tensors
        all_embeddings: List[torch.Tensor] = []
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            batch_docs = documents[start:end]

            logger.info(
                f"  Encoding documents [{start}:{end}] / {len(documents)}"
            )
            embeddings = self.model.encode(
                batch_docs,
                batch_size=self.encode_batch_size,
                is_query=False,
                show_progress_bar=True,
            )
            all_embeddings.extend(_to_tensors(embeddings))

        # Step 2: Create index with ALL documents at once
        logger.info(
            f"Building fast-plaid index with {len(all_embeddings)} documents..."
        )
        self.fp_index = fp_search.FastPlaid(
            index=index_path, device=self.device
        )
        self.fp_index.create(documents_embeddings=all_embeddings)

        logger.info(
            f"Document indexing complete. "
            f"Total: {self.total_docs} documents indexed."
        )

    def retrieve(
        self,
        queries: List[str],
        query_ids: List[str],
        top_k: int = 200,
        batch_size: int = 500,
        search_batch_size: int = 25000,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve top-K documents for each query.

        Args:
            queries: List of query texts.
            query_ids: List of query IDs.
            top_k: Number of top results to retrieve per query.
            batch_size: Batch size for encoding queries.
            search_batch_size: Internal batch size for fast-plaid search.

        Returns:
            Dict mapping query_id -> list of (document_id, score) sorted by
            score descending.
        """
        if self.fp_index is None:
            raise RuntimeError(
                "Index not built. Call encode_and_index_documents first."
            )

        logger.info(f"Retrieving top-{top_k} for {len(queries)} queries...")

        all_results: Dict[str, List[Tuple[str, float]]] = {}

        for start in range(0, len(queries), batch_size):
            end = min(start + batch_size, len(queries))
            batch_queries = queries[start:end]
            batch_qids = query_ids[start:end]

            logger.info(
                f"  Retrieving for queries [{start}:{end}] / {len(queries)}"
            )
            embeddings = self.model.encode(
                batch_queries,
                batch_size=self.encode_batch_size,
                is_query=True,
                show_progress_bar=True,
            )
            tensor_embs = _to_tensors(embeddings)
            queries_batch = _stack_queries(tensor_embs)

            # Use n_full_scores slightly above top_k for speed.
            # n_ivf_probe=2 reduces cluster probes (default=8) for faster search.
            n_full_scores = max(top_k * 4, 256)
            results = self.fp_index.search(
                queries_embeddings=queries_batch,
                top_k=top_k,
                batch_size=search_batch_size,
                n_full_scores=n_full_scores,
                n_ivf_probe=2,
            )

            for qid, result_list in zip(batch_qids, results):
                all_results[qid] = [
                    (self.doc_id_mapping[doc_idx], score)
                    for doc_idx, score in result_list
                ]

        logger.info("Retrieval complete.")
        return all_results
