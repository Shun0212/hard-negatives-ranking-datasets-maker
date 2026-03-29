"""Post-hoc deduplication of already-uploaded KD datasets on HuggingFace Hub.

Downloads documents/scores subsets, merges documents with identical text,
remaps document_ids in scores, and re-uploads.

Usage:
    uv run python deduplicate_uploaded.py --repo Shuu12121/coir_hard_negative_datasets_kd
    uv run python deduplicate_uploaded.py --repo Shuu12121/coir_hard_negative_datasets_kd --targets cosqa-python codetrans-dl-mixed
    uv run python deduplicate_uploaded.py --repo Shuu12121/coir_hard_negative_datasets_kd --dry-run
"""

import argparse
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import datasets as hf_datasets
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configs known to have significant corpus duplicates
DEFAULT_TARGETS = [
    "cosqa-python",
    "codetrans-dl-mixed",
    "codefeedback-st-mixed",
    "synthetic-text2sql-sql",
    "codesearchnet-ccr-go",
    "codesearchnet-ccr-java",
    "codesearchnet-ccr-javascript",
    "codesearchnet-ccr-php",
    "codesearchnet-ccr-python",
    "codesearchnet-ccr-ruby",
]


def deduplicate_documents(
    docs_ds: hf_datasets.Dataset,
) -> Tuple[hf_datasets.Dataset, Dict[str, str]]:
    """Deduplicate documents by text content.

    Returns:
        - Deduplicated Dataset
        - Mapping of old_id -> canonical_id for all duplicate IDs
    """
    text_to_canonical: Dict[str, str] = {}
    old_to_canonical: Dict[str, str] = {}

    for row in docs_ds:
        did = row["document_id"]
        text = row["document"]
        if text not in text_to_canonical:
            text_to_canonical[text] = did
        old_to_canonical[did] = text_to_canonical[text]

    # Build deduplicated data
    seen: set = set()
    new_data = {"document_id": [], "document": [], "split": []}
    for row in docs_ds:
        text = row["document"]
        if text not in seen:
            seen.add(text)
            new_data["document_id"].append(text_to_canonical[text])
            new_data["document"].append(text)
            new_data["split"].append(row["split"])

    return hf_datasets.Dataset.from_dict(new_data), old_to_canonical


def remap_scores(
    scores_ds: hf_datasets.Dataset,
    id_map: Dict[str, str],
) -> hf_datasets.Dataset:
    """Remap document_ids in scores and deduplicate within each row."""
    new_data = {
        "query_id": [],
        "document_ids": [],
        "scores": [],
        "split": [],
    }

    for row in scores_ds:
        doc_ids: List[str] = row["document_ids"]
        scores: List[float] = row["scores"]

        # Remap and deduplicate, keeping first occurrence (highest priority)
        seen_ids: OrderedDict = OrderedDict()
        for did, score in zip(doc_ids, scores):
            canonical = id_map.get(did, did)
            if canonical not in seen_ids:
                seen_ids[canonical] = score

        new_data["query_id"].append(row["query_id"])
        new_data["document_ids"].append(list(seen_ids.keys()))
        new_data["scores"].append(list(seen_ids.values()))
        new_data["split"].append(row["split"])

    return hf_datasets.Dataset.from_dict(new_data)


def process_config(repo_id: str, config_name: str, dry_run: bool = False) -> None:
    """Deduplicate a single config's documents and scores."""
    docs_config = f"documents_{config_name}"
    scores_config = f"scores_{config_name}"

    logger.info(f"Loading {docs_config}...")
    docs_ds = hf_datasets.load_dataset(repo_id, docs_config, split="train")
    logger.info(f"Loading {scores_config}...")
    scores_ds = hf_datasets.load_dataset(repo_id, scores_config, split="train")

    original_count = len(docs_ds)
    deduped_docs, id_map = deduplicate_documents(docs_ds)
    new_count = len(deduped_docs)
    removed = original_count - new_count

    if removed == 0:
        logger.info(f"  {config_name}: No duplicates found, skipping")
        return

    pct = removed / original_count * 100
    logger.info(
        f"  {config_name}: {original_count} -> {new_count} documents "
        f"({removed} removed, {pct:.1f}%)"
    )

    deduped_scores = remap_scores(scores_ds, id_map)

    # Check how many doc_ids were remapped in scores
    total_remapped = sum(
        1 for did, canonical in id_map.items() if did != canonical
    )
    logger.info(f"  {total_remapped} document IDs remapped to canonical IDs")

    if dry_run:
        logger.info(f"  [DRY RUN] Would upload {docs_config} and {scores_config}")
        return

    logger.info(f"  Uploading {docs_config}...")
    deduped_docs.push_to_hub(repo_id, config_name=docs_config, split="train")
    logger.info(f"  Uploading {scores_config}...")
    deduped_scores.push_to_hub(repo_id, config_name=scores_config, split="train")
    logger.info(f"  ✅ {config_name} done")


def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(hf_token)

    parser = argparse.ArgumentParser(
        description="Deduplicate already-uploaded KD datasets"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. Shuu12121/coir_hard_negative_datasets_kd)",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Config names to deduplicate (default: all known duplicates)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without uploading",
    )
    args = parser.parse_args()

    targets = args.targets or DEFAULT_TARGETS
    logger.info(f"Deduplicating {len(targets)} configs in {args.repo}")

    for config_name in targets:
        try:
            process_config(args.repo, config_name, dry_run=args.dry_run)
        except Exception as e:
            logger.error(f"  Failed {config_name}: {e}")

    logger.info("=== All done! ===")


if __name__ == "__main__":
    main()
