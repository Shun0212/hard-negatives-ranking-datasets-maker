"""YAML config loading with dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str
    languages: List[str]
    query_field: str
    documents_field: str
    lang_as_config: bool = False
    dataset_type: str = "paired"  # "paired" or "coir"
    split: str = "train"
    # CoIR-specific fields
    corpus_config: str = "corpus"
    queries_config: str = "queries"
    qrels_config: str = "default"
    corpus_split: str = "corpus"    # Split name for corpus dataset
    queries_split: str = "queries"  # Split name for queries dataset
    qrels_split: str = ""           # Split name for qrels (empty = use 'split' field)
    lang_config_format: str = "{config}_{lang}"  # Template for per-language config names
    corpus_text_field: str = "text"
    corpus_id_field: str = "_id"
    queries_text_field: str = "text"
    queries_id_field: str = "_id"
    qrels_query_id_field: str = "query-id"
    qrels_corpus_id_field: str = "corpus-id"
    qrels_score_field: str = "score"
    strip_docstrings: bool = False  # Remove docstrings from Python code documents
    deduplicate_corpus: bool = False  # Merge documents with identical text into one ID


@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""

    name: str
    architecture: str = "colbert"
    max_seq_length: int = 0  # 0 = use model default


@dataclass
class FaissConfig:
    """FAISS-specific index configuration."""

    index_type: str = "flat"  # "flat", "ivf", "ivfpq", "hnsw"
    metric: str = "cosine"  # "cosine", "ip", "l2"
    nlist: int = 100  # Number of IVF clusters
    nprobe: int = 10  # Number of clusters to probe at search time
    m_pq: int = 8  # Number of PQ sub-quantizers (for ivfpq)
    hnsw_m: int = 32  # HNSW graph degree
    ef_search: int = 64  # HNSW search depth
    use_gpu: bool = False  # Move FAISS index to GPU


@dataclass
class MiningConfig:
    """Configuration for hard negative mining."""

    top_k: int = 200
    encode_batch_size: int = 32
    index_batch_size: int = 500
    query_batch_size: int = 500
    search_batch_size: int = 25000
    nv_threshold: float = 0.95
    num_negatives: int = 100
    index_dir: str = "./plaid_index"
    faiss_index_dir: str = "./faiss_index"
    cache_dir: str = "./cache"
    device: str = ""  # "" = auto-detect (cuda if available, else cpu)
    faiss: FaissConfig = field(default_factory=FaissConfig)


@dataclass
class UploadConfig:
    """Configuration for uploading to HuggingFace Hub."""

    dataset: str = ""
    max_per_language: int = 100000
    max_per_query: int = 100


@dataclass
class ProjectConfig:
    """Top-level project configuration."""

    datasets: List[DatasetConfig]
    embedding_models: List[EmbeddingModelConfig]
    mining_config: MiningConfig
    upload_config: UploadConfig


def load_config(path: str) -> ProjectConfig:
    """Load project config from a YAML file."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    datasets = [DatasetConfig(**d) for d in raw["datasets"]]
    models = [EmbeddingModelConfig(**m) for m in raw["embedding_models"]]

    mining_raw = raw.get("mining_config", {})
    faiss_raw = mining_raw.pop("faiss", {})
    mining = MiningConfig(**mining_raw, faiss=FaissConfig(**faiss_raw))

    upload_raw = raw.get("upload_config", {})
    # Handle the list-of-single-key-dicts format (original config style)
    if isinstance(upload_raw, list):
        merged = {}
        for item in upload_raw:
            merged.update(item)
        upload_raw = merged
    upload = UploadConfig(**upload_raw)

    return ProjectConfig(
        datasets=datasets,
        embedding_models=models,
        mining_config=mining,
        upload_config=upload,
    )
