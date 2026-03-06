"""Abstract base class for encoder backends."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseEncoder(ABC):
    """Encoder interface for hard negative mining.

    Implementations must provide:
    - encode_and_index_documents(): encode a corpus and build a search index
    - retrieve(): search the index for top-K results per query
    """

    @abstractmethod
    def encode_and_index_documents(
        self,
        documents: List[str],
        document_ids: List[str],
        index_name: str = "documents",
        batch_size: int = 500,
    ) -> None:
        ...

    @abstractmethod
    def retrieve(
        self,
        queries: List[str],
        query_ids: List[str],
        top_k: int = 200,
        batch_size: int = 500,
        search_batch_size: int = 25000,
    ) -> Dict[str, List[Tuple[str, float]]]:
        ...
