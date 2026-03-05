"""Upload datasets to HuggingFace Hub."""

import logging

import datasets as hf_datasets

logger = logging.getLogger(__name__)


def upload_dataset(
    dataset: hf_datasets.Dataset,
    repo_id: str,
    config_name: str = "default",
    split: str = "train",
    private: bool = False,
) -> None:
    """Upload a single Dataset to HuggingFace Hub."""
    logger.info(f"Uploading to {repo_id} (config={config_name}, split={split})...")
    dataset.push_to_hub(
        repo_id,
        config_name=config_name,
        split=split,
        private=private,
    )
    logger.info(f"Upload complete: {repo_id}/{config_name}")


def upload_kd_dataset(
    kd_datasets: dict[str, hf_datasets.Dataset],
    repo_id: str,
    language: str,
    private: bool = False,
) -> None:
    """Upload KD-format datasets (queries, documents, scores) to Hub.

    Each subset is uploaded as a separate config with language suffix.
    """
    for subset_name, ds in kd_datasets.items():
        config_name = f"{subset_name}_{language}" if language else subset_name
        upload_dataset(ds, repo_id, config_name=config_name, private=private)
