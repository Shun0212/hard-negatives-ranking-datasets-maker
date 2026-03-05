"""Dataset loaders for paired (docstring-code) and CoIR (BEIR-format) datasets."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import datasets as hf_datasets

from .config import DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Unified dataset representation for hard negative mining.

    Holds queries, documents, and their relevance relationships (qrels)
    in a format that works for both paired and CoIR datasets.
    """

    queries: List[str]
    query_ids: List[str]
    documents: List[str]
    document_ids: List[str]
    qrels: Dict[str, List[str]]  # query_id -> list of positive document_ids
    language: str
    dataset_name: str


def load_paired_dataset(config: DatasetConfig) -> List[DatasetBundle]:
    """Load a paired dataset where each row is a (query, document) pair.

    For example, datasets with (docstring, code) pairs. Each query's
    positive document is the document in the same row.
    """
    bundles = []
    for lang in config.languages:
        logger.info(f"Loading paired dataset: {config.name} (lang={lang})")

        if config.lang_as_config:
            ds = hf_datasets.load_dataset(config.name, lang, split=config.split)
        else:
            ds = hf_datasets.load_dataset(config.name, split=config.split)

        queries = []
        query_ids = []
        documents = []
        document_ids = []
        qrels: Dict[str, List[str]] = {}

        for i, row in enumerate(ds):
            query_text = row.get(config.query_field)
            doc_text = row.get(config.documents_field)

            if not query_text or not doc_text:
                continue

            qid = f"q_{lang}_{i}"
            did = f"d_{lang}_{i}"

            queries.append(str(query_text))
            query_ids.append(qid)
            documents.append(str(doc_text))
            document_ids.append(did)
            qrels[qid] = [did]

        logger.info(
            f"  Loaded {len(queries)} query-document pairs for {lang}"
        )

        bundles.append(
            DatasetBundle(
                queries=queries,
                query_ids=query_ids,
                documents=documents,
                document_ids=document_ids,
                qrels=qrels,
                language=lang,
                dataset_name=config.name,
            )
        )

    return bundles


def load_coir_dataset(config: DatasetConfig) -> List[DatasetBundle]:
    """Load a CoIR (BEIR-format) dataset with separate corpus, queries, and qrels.

    CoIR datasets typically have:
    - corpus config: _id, text, (optional) title
    - queries config: _id, text
    - qrels config: query-id, corpus-id, score
    """
    bundles = []
    for lang in config.languages:
        logger.info(f"Loading CoIR dataset: {config.name} (lang={lang})")

        # Build config names, optionally appending language
        if config.lang_as_config:
            corpus_cfg = f"{config.corpus_config}_{lang}"
            queries_cfg = f"{config.queries_config}_{lang}"
            qrels_cfg = f"{config.qrels_config}_{lang}"
        else:
            corpus_cfg = config.corpus_config
            queries_cfg = config.queries_config
            qrels_cfg = config.qrels_config

        corpus_ds = hf_datasets.load_dataset(
            config.name, corpus_cfg, split=config.split
        )
        queries_ds = hf_datasets.load_dataset(
            config.name, queries_cfg, split=config.split
        )
        qrels_ds = hf_datasets.load_dataset(
            config.name, qrels_cfg, split=config.split
        )

        # Build documents
        documents = []
        document_ids = []
        for row in corpus_ds:
            did = str(row[config.corpus_id_field])
            text = str(row.get(config.corpus_text_field, ""))
            title = str(row.get("title", ""))
            doc_text = f"{title} {text}".strip() if title else text
            if not doc_text:
                continue
            documents.append(doc_text)
            document_ids.append(did)

        # Build queries
        queries = []
        query_ids = []
        for row in queries_ds:
            qid = str(row[config.queries_id_field])
            text = str(row[config.queries_text_field])
            if not text:
                continue
            queries.append(text)
            query_ids.append(qid)

        # Build qrels (only positive relevance)
        qrels: Dict[str, List[str]] = {}
        for row in qrels_ds:
            qid = str(row[config.qrels_query_id_field])
            did = str(row[config.qrels_corpus_id_field])
            score = row.get(config.qrels_score_field, 1)
            if score > 0:
                qrels.setdefault(qid, []).append(did)

        logger.info(
            f"  Loaded {len(queries)} queries, {len(documents)} documents, "
            f"{len(qrels)} queries with relevance judgments for {lang}"
        )

        bundles.append(
            DatasetBundle(
                queries=queries,
                query_ids=query_ids,
                documents=documents,
                document_ids=document_ids,
                qrels=qrels,
                language=lang,
                dataset_name=config.name,
            )
        )

    return bundles


def load_dataset_bundle(config: DatasetConfig) -> List[DatasetBundle]:
    """Load a dataset based on its type (paired or coir)."""
    if config.dataset_type == "coir":
        return load_coir_dataset(config)
    else:
        return load_paired_dataset(config)
