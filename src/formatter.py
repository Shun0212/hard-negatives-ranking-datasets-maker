"""Format mining results into KD and contrastive datasets."""

import logging
from typing import Dict, List

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

    Filters hard negatives by score threshold (NV-Retriever style):
    a negative is valid if score < nv_threshold * positive_score.
    Only queries with enough valid negatives are included.

    Output columns: query, positive, negative_0, negative_1, ..., negative_N
    """

    def __init__(
        self,
        kd_results: List[KDResult],
        bundle: DatasetBundle,
        num_negatives: int = 32,
        nv_threshold: float = 0.95,
        max_per_language: int = 100000,
    ):
        self.kd_results = kd_results
        self.query_id_to_text = dict(zip(bundle.query_ids, bundle.queries))
        self.doc_id_to_text = dict(zip(bundle.document_ids, bundle.documents))
        self.num_negatives = num_negatives
        self.nv_threshold = nv_threshold
        self.max_per_language = max_per_language

    def has_enough_negatives(self, kd_result: KDResult) -> bool:
        """Check if a query has at least num_negatives valid hard negatives."""
        scores = kd_result.scores
        if len(scores) < 2:
            return False
        positive_score = scores[0]
        if positive_score <= 0:
            return False
        count = sum(
            1
            for score in scores[1:]
            if score < self.nv_threshold * positive_score and score != -1
        )
        return count >= self.num_negatives

    def convert(self) -> hf_datasets.Dataset:
        """Convert KD results to contrastive format Dataset.

        Each row: query, positive, negative_0, ..., negative_{num_negatives-1}
        """
        rows: List[Dict[str, str]] = []

        for kd_result in self.kd_results:
            if len(rows) >= self.max_per_language:
                break

            if not self.has_enough_negatives(kd_result):
                continue

            query_text = self.query_id_to_text[kd_result.query_id]
            positive_id = kd_result.document_ids[0]
            positive_text = self.doc_id_to_text[positive_id]
            positive_score = kd_result.scores[0]

            row: Dict[str, str] = {
                "query": query_text,
                "positive": positive_text,
            }

            neg_count = 0
            for i in range(1, len(kd_result.document_ids)):
                score = kd_result.scores[i]
                if score < self.nv_threshold * positive_score and score != -1:
                    doc_id = kd_result.document_ids[i]
                    row[f"negative_{neg_count}"] = self.doc_id_to_text[doc_id]
                    neg_count += 1
                    if neg_count >= self.num_negatives:
                        break

            rows.append(row)

        if not rows:
            logger.warning("No queries passed the threshold filter")
            return hf_datasets.Dataset.from_dict({})

        # Ensure all rows have the same columns (pad missing negatives)
        for row in rows:
            for i in range(self.num_negatives):
                row.setdefault(f"negative_{i}", "")

        # Build column-oriented dict for Dataset creation
        columns = list(rows[0].keys())
        data = {col: [row[col] for row in rows] for col in columns}

        logger.info(
            f"Converted {len(rows)} queries to contrastive format "
            f"({self.num_negatives} negatives each)"
        )
        return hf_datasets.Dataset.from_dict(data)
