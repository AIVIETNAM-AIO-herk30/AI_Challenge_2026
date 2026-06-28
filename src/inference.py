"""
End-to-end inference: query → classify → dispatch → retrieve video frames.
Owner: Truong Hoang Thong
"""

import argparse

import yaml


def search(query: str, config: dict, top_k: int = 10) -> list[dict]:
    """
    TODO (Truong Hoang Thong):
    - Load QueryClassifier from weights
    - Classify the query (type + complexity)
    - Pass to DynamicDispatcher to get agent results
    - Embed the query via VisualAgent (text embedding)
    - Search VectorStore and return top_k hits

    Returns: list of {"video_id": str, "timestamp_sec": float, "score": float}
    """
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = search(args.query, cfg, args.top_k)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")
