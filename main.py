"""Hard Negative Mining with ColBERT via PyLate.

Usage:
    uv run python main.py --config config/docstring_to_code.yaml --output-format both --save-local ./output
    uv run python main.py --config config/docstring_to_code.yaml --upload
"""

import argparse
import logging
import os

from dotenv import load_dotenv

from src.config import load_config
from src.data_loader import load_dataset_bundle
from src.encoder import ColBERTEncoder
from src.formatter import KDToContrastive, build_kd_dataset
from src.miner import HardNegativeMiner
from src.uploader import upload_dataset, upload_kd_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # Load environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)
        logger.info("HuggingFace login successful")
    else:
        logger.warning("HF_TOKEN not set")

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Mine hard negatives using ColBERT (PyLate)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output-format",
        choices=["kd", "contrastive", "both"],
        default="both",
        help="Output format: kd (knowledge distillation), contrastive, or both",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload results to HuggingFace Hub",
    )
    parser.add_argument(
        "--save-local",
        type=str,
        default=None,
        help="Save datasets locally to this directory",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    model_config = config.embedding_models[0]

    # Initialize encoder
    encoder = ColBERTEncoder(
        model_name=model_config.name,
        index_dir=config.mining_config.index_dir,
        encode_batch_size=config.mining_config.encode_batch_size,
        device=config.mining_config.device or None,
    )

    # Initialize miner
    miner = HardNegativeMiner(encoder=encoder, config=config.mining_config)

    # Process each dataset
    for ds_config in config.datasets:
        logger.info(f"=== Processing dataset: {ds_config.name} ===")
        bundles = load_dataset_bundle(ds_config)

        for bundle in bundles:
            logger.info(f"--- Language: {bundle.language} ---")

            # Mine hard negatives
            kd_results, bundle = miner.mine(bundle)

            if not kd_results:
                logger.warning(f"No results for {bundle.language}, skipping")
                continue

            # Build and save/upload KD format
            if args.output_format in ("kd", "both"):
                kd_datasets = build_kd_dataset(
                    kd_results, bundle, split_name=bundle.language
                )

                if args.save_local:
                    for name, ds in kd_datasets.items():
                        path = os.path.join(
                            args.save_local, "kd", bundle.language, name
                        )
                        ds.save_to_disk(path)
                        logger.info(f"Saved KD/{name} to {path}")

                if args.upload:
                    upload_kd_dataset(
                        kd_datasets,
                        config.upload_config.dataset + "_kd",
                        language=bundle.language,
                    )

            # Build and save/upload contrastive format
            if args.output_format in ("contrastive", "both"):
                converter = KDToContrastive(
                    kd_results=kd_results,
                    bundle=bundle,
                    num_negatives=config.upload_config.max_per_query,
                    nv_threshold=config.mining_config.nv_threshold,
                    max_per_language=config.upload_config.max_per_language,
                )
                contrastive_ds = converter.convert()

                if len(contrastive_ds) == 0:
                    logger.warning(
                        f"No queries passed threshold for {bundle.language}"
                    )
                    continue

                logger.info(
                    f"Contrastive dataset: {len(contrastive_ds)} rows "
                    f"for {bundle.language}"
                )

                if args.save_local:
                    path = os.path.join(
                        args.save_local, "contrastive", bundle.language
                    )
                    contrastive_ds.save_to_disk(path)
                    logger.info(f"Saved contrastive to {path}")

                if args.upload:
                    upload_dataset(
                        contrastive_ds,
                        config.upload_config.dataset,
                        config_name=bundle.language,
                    )

    logger.info("=== All done! ===")


if __name__ == "__main__":
    main()
