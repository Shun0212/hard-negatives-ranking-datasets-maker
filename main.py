"""Hard Negative Mining with ColBERT or Sentence-Transformers.

Usage:
    uv run python main.py --config config/docstring_to_code.yaml --output-format both --save-local ./output
    uv run python main.py --config config\docstring_to_code-multi-lang-codesearch-query-qwen.yaml --upload
"""

import argparse
import logging
import os

from dotenv import load_dotenv

from src.base_encoder import BaseEncoder
from src.config import EmbeddingModelConfig, MiningConfig, load_config
from src.data_loader import load_dataset_bundle
from src.formatter import KDToContrastive, build_kd_dataset
from src.miner import HardNegativeMiner
from src.uploader import upload_dataset, upload_kd_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_encoder(
    model_config: EmbeddingModelConfig,
    mining_config: MiningConfig,
) -> BaseEncoder:
    """Create the appropriate encoder based on architecture config."""
    arch = model_config.architecture

    if arch == "colbert":
        from src.encoder import ColBERTEncoder

        return ColBERTEncoder(
            model_name=model_config.name,
            index_dir=mining_config.index_dir,
            encode_batch_size=mining_config.encode_batch_size,
            device=mining_config.device or None,
        )
    elif arch == "sentence-transformers":
        from src.sentence_transformer_encoder import SentenceTransformerEncoder

        faiss_cfg = mining_config.faiss
        return SentenceTransformerEncoder(
            model_name=model_config.name,
            index_dir=mining_config.faiss_index_dir,
            encode_batch_size=mining_config.encode_batch_size,
            device=mining_config.device or None,
            max_seq_length=model_config.max_seq_length,
            faiss_index_type=faiss_cfg.index_type,
            faiss_metric=faiss_cfg.metric,
            faiss_nlist=faiss_cfg.nlist,
            faiss_nprobe=faiss_cfg.nprobe,
            faiss_m_pq=faiss_cfg.m_pq,
            faiss_hnsw_m=faiss_cfg.hnsw_m,
            faiss_ef_search=faiss_cfg.ef_search,
            faiss_use_gpu=faiss_cfg.use_gpu,
        )
    else:
        raise ValueError(
            f"Unknown architecture: {arch}. "
            "Use 'colbert' or 'sentence-transformers'."
        )


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
    encoder = create_encoder(model_config, config.mining_config)

    # Initialize miner
    miner = HardNegativeMiner(encoder=encoder, config=config.mining_config)

    # Process each dataset
    for ds_config in config.datasets:
        logger.info(f"=== Processing dataset: {ds_config.name} ===")
        bundles = load_dataset_bundle(ds_config)

        # Derive a short task prefix for config naming (e.g. "codesearchnet")
        task_prefix = ds_config.name.split("/")[-1].lower()

        for bundle in bundles:
            logger.info(f"--- Language: {bundle.language} ---")

            # Build config_name: for CoIR use "task-lang", for paired use "lang"
            if ds_config.dataset_type == "coir":
                config_name = f"{task_prefix}-{bundle.language}"
            else:
                config_name = bundle.language

            # Mine hard negatives
            kd_results, bundle = miner.mine(
                bundle,
                max_queries=config.upload_config.max_per_language,
            )

            if not kd_results:
                logger.warning(f"No results for {config_name}, skipping")
                continue

            # Build and save/upload KD format
            if args.output_format in ("kd", "both"):
                kd_datasets = build_kd_dataset(
                    kd_results, bundle, split_name=bundle.language
                )

                if args.save_local:
                    for name, ds in kd_datasets.items():
                        path = os.path.join(
                            args.save_local, "kd", config_name, name
                        )
                        ds.save_to_disk(path)
                        logger.info(f"Saved KD/{name} to {path}")

                if args.upload:
                    upload_kd_dataset(
                        kd_datasets,
                        config.upload_config.dataset + "_kd",
                        language=config_name,
                    )

            # Build and save/upload contrastive format (all negatives with scores, no threshold)
            if args.output_format in ("contrastive", "both"):
                converter = KDToContrastive(
                    kd_results=kd_results,
                    bundle=bundle,
                    num_negatives=config.upload_config.max_per_query,
                    max_per_language=config.upload_config.max_per_language,
                )
                contrastive_ds = converter.convert()

                if len(contrastive_ds) == 0:
                    logger.warning(
                        f"No valid queries for {config_name}"
                    )
                    continue

                logger.info(
                    f"Contrastive dataset: {len(contrastive_ds)} rows "
                    f"for {config_name}"
                )

                if args.save_local:
                    path = os.path.join(
                        args.save_local, "contrastive", config_name
                    )
                    contrastive_ds.save_to_disk(path)
                    logger.info(f"Saved contrastive to {path}")

                if args.upload:
                    upload_dataset(
                        contrastive_ds,
                        config.upload_config.dataset,
                        config_name=config_name,
                    )

    logger.info("=== All done! ===")


if __name__ == "__main__":
    main()
