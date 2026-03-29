"""Dataset loaders for paired (docstring-code) and CoIR (BEIR-format) datasets."""

import ast
import logging
from dataclasses import dataclass, field
from typing import Dict, List

import datasets as hf_datasets

from .config import DatasetConfig

logger = logging.getLogger(__name__)


def _strip_python_docstrings(code: str) -> str:
    """Remove all docstrings from Python source code using the ast module.

    Handles module-level, class-level, and function/method-level docstrings.
    Returns the original code unchanged if parsing fails.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    # Collect line ranges of docstring nodes to remove
    # Docstrings are Expr nodes containing a Constant(str) as the first
    # statement of a module, class, or function body.
    remove_lines: list[tuple[int, int]] = []

    def _check_docstring(body: list[ast.stmt]) -> None:
        if not body:
            return
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            remove_lines.append((first.lineno, first.end_lineno))

    _check_docstring(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _check_docstring(node.body)

    if not remove_lines:
        return code

    lines = code.splitlines(keepends=True)
    # Build set of 1-indexed line numbers to remove
    remove_set: set[int] = set()
    for start, end in remove_lines:
        for ln in range(start, (end or start) + 1):
            remove_set.add(ln)

    result = []
    for i, line in enumerate(lines, start=1):
        if i not in remove_set:
            result.append(line)

    return "".join(result)


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


def _deduplicate_bundle(bundle: DatasetBundle) -> DatasetBundle:
    """Merge documents with identical text into a single canonical ID.

    Keeps the first-seen ID for each unique text, removes duplicate entries
    from documents/document_ids, and remaps qrels to use canonical IDs.
    """
    text_to_canonical: Dict[str, str] = {}
    old_to_canonical: Dict[str, str] = {}

    for did, text in zip(bundle.document_ids, bundle.documents):
        if text not in text_to_canonical:
            text_to_canonical[text] = did
        old_to_canonical[did] = text_to_canonical[text]

    # Build deduplicated document lists
    seen_texts: set = set()
    new_documents: List[str] = []
    new_document_ids: List[str] = []
    for did, text in zip(bundle.document_ids, bundle.documents):
        if text not in seen_texts:
            seen_texts.add(text)
            new_documents.append(text)
            new_document_ids.append(text_to_canonical[text])

    # Remap qrels to canonical IDs and deduplicate
    new_qrels: Dict[str, List[str]] = {}
    for qid, dids in bundle.qrels.items():
        remapped = list(dict.fromkeys(old_to_canonical.get(d, d) for d in dids))
        new_qrels[qid] = remapped

    removed = len(bundle.documents) - len(new_documents)
    if removed > 0:
        logger.info(
            f"  Deduplicated corpus: {len(bundle.documents)} -> {len(new_documents)} "
            f"documents ({removed} duplicates removed)"
        )

    return DatasetBundle(
        queries=bundle.queries,
        query_ids=bundle.query_ids,
        documents=new_documents,
        document_ids=new_document_ids,
        qrels=new_qrels,
        language=bundle.language,
        dataset_name=bundle.dataset_name,
    )


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

            doc_text = str(doc_text)
            if config.strip_docstrings and lang == "python":
                doc_text = _strip_python_docstrings(doc_text)

            qid = f"q_{lang}_{i}"
            did = f"d_{lang}_{i}"

            queries.append(str(query_text))
            query_ids.append(qid)
            documents.append(doc_text)
            document_ids.append(did)
            qrels[qid] = [did]

        logger.info(
            f"  Loaded {len(queries)} query-document pairs for {lang}"
        )

        bundle = DatasetBundle(
            queries=queries,
            query_ids=query_ids,
            documents=documents,
            document_ids=document_ids,
            qrels=qrels,
            language=lang,
            dataset_name=config.name,
        )
        if config.deduplicate_corpus:
            bundle = _deduplicate_bundle(bundle)
        bundles.append(bundle)

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
            corpus_cfg = config.lang_config_format.format(config=config.corpus_config, lang=lang)
            queries_cfg = config.lang_config_format.format(config=config.queries_config, lang=lang)
            qrels_cfg = config.lang_config_format.format(config=config.qrels_config, lang=lang)
        else:
            corpus_cfg = config.corpus_config
            queries_cfg = config.queries_config
            qrels_cfg = config.qrels_config

        # Determine splits for each subset
        corpus_split = config.corpus_split
        queries_split = config.queries_split
        qrels_split = config.qrels_split or config.split

        corpus_ds = hf_datasets.load_dataset(
            config.name, corpus_cfg, split=corpus_split
        )
        queries_ds = hf_datasets.load_dataset(
            config.name, queries_cfg, split=queries_split
        )
        qrels_ds = hf_datasets.load_dataset(
            config.name, qrels_cfg, split=qrels_split
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

        bundle = DatasetBundle(
            queries=queries,
            query_ids=query_ids,
            documents=documents,
            document_ids=document_ids,
            qrels=qrels,
            language=lang,
            dataset_name=config.name,
        )
        if config.deduplicate_corpus:
            bundle = _deduplicate_bundle(bundle)
        bundles.append(bundle)

    return bundles


def load_dataset_bundle(config: DatasetConfig) -> List[DatasetBundle]:
    """Load a dataset based on its type (paired or coir)."""
    if config.dataset_type == "coir":
        return load_coir_dataset(config)
    else:
        return load_paired_dataset(config)
