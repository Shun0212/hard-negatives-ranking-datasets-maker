"""Hard negative mining pipeline using ColBERT retrieval."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .config import MiningConfig
from .data_loader import DatasetBundle
from .encoder import ColBERTEncoder

logger = logging.getLogger(__name__)


@dataclass
class KDResult:
    """Knowledge distillation result for a single query.

    document_ids[0] and scores[0] correspond to the positive document.
    The remaining entries are negative candidates sorted by score descending.
    A score of -1 means the positive wasn't found in retrieval results.
    """

    query_id: str
    document_ids: List[str]
    scores: List[float]


class HardNegativeMiner:
    """Mine hard negatives using ColBERT retrieval via PyLate."""

    def __init__(self, encoder: ColBERTEncoder, config: MiningConfig):
        self.encoder = encoder
        self.config = config

    def mine(self, bundle: DatasetBundle) -> Tuple[List[KDResult], DatasetBundle]:
        """Mine hard negatives for a dataset bundle.

        1. Encode and index all documents
        2. Retrieve top-K candidates for each query
        3. Separate positives from negatives
        4. Build KD-format results

        Returns:
            Tuple of (list of KDResult, the original DatasetBundle).
        """
        logger.info(
            f"Mining hard negatives for {bundle.language}: "
            f"{len(bundle.queries)} queries, {len(bundle.documents)} documents"
        )

        # Step 1: Encode and index documents
        index_name = f"{bundle.language}_{abs(hash(bundle.dataset_name)) % 100000}"
        self.encoder.encode_and_index_documents(
            documents=bundle.documents,
            document_ids=bundle.document_ids,
            index_name=index_name,
            batch_size=self.config.index_batch_size,
        )

        # Step 2: Retrieve top-K for each query
        retrieval_results = self.encoder.retrieve(
            queries=bundle.queries,
            query_ids=bundle.query_ids,
            top_k=self.config.top_k,
            batch_size=self.config.query_batch_size,
        )

        # Step 3: Build KD results
        kd_results = []
        skipped_no_results = 0
        positive_not_retrieved = 0

        for qid in bundle.query_ids:
            if qid not in retrieval_results:
                skipped_no_results += 1
                continue

            results = retrieval_results[qid]
            positive_ids = set(bundle.qrels.get(qid, []))

            # Separate positives and negatives from retrieval results
            positive_entries: List[Tuple[str, float]] = []
            negative_entries: List[Tuple[str, float]] = []

            for doc_id, score in results:
                if doc_id in positive_ids:
                    positive_entries.append((doc_id, score))
                else:
                    negative_entries.append((doc_id, score))

            # If the positive wasn't retrieved, add with marker score
            retrieved_pos_ids = {pid for pid, _ in positive_entries}
            for pid in positive_ids:
                if pid not in retrieved_pos_ids:
                    positive_entries.append((pid, -1.0))
                    positive_not_retrieved += 1

            if not positive_entries:
                skipped_no_results += 1
                continue

            # Build result: positives first, then negatives by score descending
            negative_entries.sort(key=lambda x: x[1], reverse=True)

            doc_ids = (
                [did for did, _ in positive_entries]
                + [did for did, _ in negative_entries[: self.config.num_negatives]]
            )
            scores = (
                [s for _, s in positive_entries]
                + [s for _, s in negative_entries[: self.config.num_negatives]]
            )

            kd_results.append(
                KDResult(
                    query_id=qid,
                    document_ids=doc_ids,
                    scores=scores,
                )
            )

        if skipped_no_results:
            logger.warning(
                f"Skipped {skipped_no_results} queries with no retrieval results"
            )
        if positive_not_retrieved:
            logger.warning(
                f"{positive_not_retrieved} positives not found in top-{self.config.top_k} retrieval"
            )
        logger.info(f"Generated KD results for {len(kd_results)} queries")

        return kd_results, bundle
