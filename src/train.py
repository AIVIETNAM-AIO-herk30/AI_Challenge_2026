"""
Training pipeline for the query complexity classifier.
Owner: Pham Viet Truong
"""

import argparse

import yaml


def train(config: dict) -> None:
    """
    TODO (Pham Viet Truong):
    - Load dataset via build_dataloaders()
    - Instantiate QueryClassifier and MultiTaskLoss from model.py
    - Training loop: forward → loss → backward → optimizer step
    - Log train/val metrics each epoch
    - Save best checkpoint to config["train"]["checkpoint_dir"]
    """
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
