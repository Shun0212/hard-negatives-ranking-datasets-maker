"""Format mining results into KD and contrastive datasets."""

import logging
import random
from typing import Any, Dict, List

import datasets as hf_datasets

from .data_loader import DatasetBundle
from .miner import KDResult

logger = logging.getLogger(__name__)


def build_kd_dataset(
    kd_results: List[KDResult],
    bundle: DatasetBundle,
    split_name: str = "train",
) -> Dict[str, hf_datasets.Dataset]:
    """Build a KD-format dataset with queries, documents, and scores subsets.

    Output structure matches lightonai/nv-embed-supervised-distill-dedup-code:
    - queries: query_id, query, split
    - documents: document_id, document, split
    - scores: query_id, document_ids, scores, split
    """
    query_id_to_text = dict(zip(bundle.query_ids, bundle.queries))
    doc_id_to_text = dict(zip(bundle.document_ids, bundle.documents))

    # Collect used IDs
    used_qids = set()
    used_dids = set()
    for r in kd_results:
        used_qids.add(r.query_id)
        for did in r.document_ids:
            used_dids.add(did)

    # Build queries subset
    queries_data = {"query_id": [], "query": [], "split": []}
    for qid in bundle.query_ids:
        if qid in used_qids:
            queries_data["query_id"].append(qid)
            queries_data["query"].append(query_id_to_text[qid])
            queries_data["split"].append(split_name)

    # Build documents subset
    documents_data = {"document_id": [], "document": [], "split": []}
    for did in bundle.document_ids:
        if did in used_dids:
            documents_data["document_id"].append(did)
            documents_data["document"].append(doc_id_to_text[did])
            documents_data["split"].append(split_name)

    # Build scores subset
    scores_data = {
        "query_id": [],
        "document_ids": [],
        "scores": [],
        "split": [],
    }
    for r in kd_results:
        scores_data["query_id"].append(r.query_id)
        scores_data["document_ids"].append(r.document_ids)
        scores_data["scores"].append(r.scores)
        scores_data["split"].append(split_name)

    return {
        "queries": hf_datasets.Dataset.from_dict(queries_data),
        "documents": hf_datasets.Dataset.from_dict(documents_data),
        "scores": hf_datasets.Dataset.from_dict(scores_data),
    }


class KDToContrastive:
    """Convert KD-format results to contrastive training format.

    Saves all negatives with their scores without threshold filtering.
    Threshold-based filtering can be applied later using the saved scores.

    Output columns:
        query, positive, positive_score,
        negative_0, negative_0_score, ..., negative_N, negative_N_score
    """

    def __init__(
        self,
        kd_results: List[KDResult],
        bundle: DatasetBundle,
        num_negatives: int = 100,
        max_per_language: int = 100000,
        seed: int = 42,
    ):
        self.kd_results = kd_results
        self.query_id_to_text = dict(zip(bundle.query_ids, bundle.queries))
        self.doc_id_to_text = dict(zip(bundle.document_ids, bundle.documents))
        self.num_negatives = num_negatives
        self.max_per_language = max_per_language
        self.seed = seed

    def convert(self) -> hf_datasets.Dataset:
        """Convert KD results to contrastive format Dataset with scores.

        Each row contains query, positive (with score), and up to
        num_negatives negatives (each with score), sorted by score descending.
        No threshold filtering is applied.
        When max_per_language < total valid queries, rows are randomly sampled.
        """
        # First pass: collect all valid rows
        all_rows: List[Dict[str, Any]] = []

        for kd_result in self.kd_results:
            # Need at least a positive and one negative
            if len(kd_result.document_ids) < 2:
                continue

            # Skip if the positive was not retrieved (score == -1)
            if kd_result.scores[0] <= 0:
                continue

            query_text = self.query_id_to_text[kd_result.query_id]
            positive_id = kd_result.document_ids[0]
            positive_text = self.doc_id_to_text[positive_id]
            positive_score = float(kd_result.scores[0])

            row: Dict[str, Any] = {
                "query": query_text,
                "positive": positive_text,
                "positive_score": positive_score,
            }

            neg_count = 0
            for i in range(1, len(kd_result.document_ids)):
                if neg_count >= self.num_negatives:
                    break
                doc_id = kd_result.document_ids[i]
                score = float(kd_result.scores[i])
                row[f"negative_{neg_count}"] = self.doc_id_to_text[doc_id]
                row[f"negative_{neg_count}_score"] = score
                neg_count += 1

            all_rows.append(row)

        if not all_rows:
            logger.warning("No valid queries to convert")
            return hf_datasets.Dataset.from_dict({})

        # Random sampling if over limit
        if len(all_rows) > self.max_per_language:
            logger.info(
                f"Randomly sampling {self.max_per_language} / {len(all_rows)} "
                f"rows (seed={self.seed})"
            )
            rng = random.Random(self.seed)
            all_rows = rng.sample(all_rows, self.max_per_language)

        # Pad missing negatives with empty string / 0.0
        for row in all_rows:
            for i in range(self.num_negatives):
                row.setdefault(f"negative_{i}", "")
                row.setdefault(f"negative_{i}_score", 0.0)

        # Build column-oriented dict for Dataset creation
        columns = list(all_rows[0].keys())
        data = {col: [row[col] for row in all_rows] for col in columns}

        logger.info(
            f"Converted {len(all_rows)} queries to contrastive format "
            f"(up to {self.num_negatives} negatives each, with scores)"
        )
        return hf_datasets.Dataset.from_dict(data)
