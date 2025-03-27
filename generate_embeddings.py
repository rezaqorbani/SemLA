from pathlib import Path
import os
import yaml
import argparse
from domain_orchestrator.utils import get_domain_args

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS")

if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser()
    # Path to the yaml file that contains the paths to the domains training data
    parser.add_argument("--source_domains_file", type=str, required=True)
    # Path to the lora library where the statistics will be stored
    parser.add_argument("--lora_library_path", type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    source_domains_file = Path(args.source_domains_file)
    lora_library_path = Path(args.lora_library_path)

    with open(source_domains_file, "r") as f:
        source_domains = yaml.safe_load(f)

    embedding_manager = None

    print("Generating embeddings for all source domains ...")
    
    for domain_name in source_domains:

        args = get_domain_args(domain_name, "train", get_cofing_only=True)
        train_dataset_path = Path(args.train_dataset_path)
        print(train_dataset_path)

        assert train_dataset_path.exists(), f"Path to training dataset {train_dataset_path} does not exist!"

        if embedding_manager is None:
            from domain_orchestrator import embedding
            embedding_manager = embedding.EmbeddingManager()

        domain_path = lora_library_path / Path(domain_name)

        embedding_manager.calculate_statistics(
            domain_name=domain_name,
            domain_path=domain_path,
            train_path=train_dataset_path,
        )

    print("Finished generating embeddings for all domains")