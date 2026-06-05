from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from src.data_wordnet import ensure_synthetic_data
from src.evaluate import aggregate_results
from src.train import run_one
from src.visualize import main as visualize_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small_experiment.yaml")
    parser.add_argument("--skip-finalize", action="store_true", help="Only run training jobs; skip aggregate tables and figures")
    parser.add_argument("--finalize-only", action="store_true", help="Only aggregate existing logs and regenerate figures")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    results_root = cfg["outputs"]["root"]
    Path(results_root, "logs").mkdir(parents=True, exist_ok=True)
    Path(results_root, "tables").mkdir(parents=True, exist_ok=True)
    Path(results_root, "figures").mkdir(parents=True, exist_ok=True)
    embeddings_path = cfg["data"].get("embeddings_by_dim", cfg["data"].get("embeddings_path"))
    if isinstance(embeddings_path, dict):
        missing = [embeddings_path[str(dim)] for dim in cfg["experiment"]["dims"] if not Path(embeddings_path[str(dim)]).exists()]
        if missing:
            raise FileNotFoundError("Missing dim-specific embeddings. Run scripts/convert_poincare_dims.sh first: " + ", ".join(missing))
        if not Path(cfg["data"]["edges_path"]).exists():
            raise FileNotFoundError(cfg["data"]["edges_path"])
    else:
        ensure_synthetic_data(embeddings_path, cfg["data"]["edges_path"], max_dim=max(cfg["experiment"]["dims"]))

    if args.finalize_only:
        aggregate_results(results_root)
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["visualize", "--config", str(config_path)]
            visualize_main()
        finally:
            sys.argv = old_argv
        return

    jobs = []
    for subtree in cfg["experiment"]["subtrees"]:
        for dim in cfg["experiment"]["dims"]:
            for seed in cfg["experiment"]["seeds"]:
                for setting in cfg["settings"]:
                    jobs.append((subtree, dim, seed, setting))
    for subtree, dim, seed, setting in tqdm(jobs, desc="experiments"):
        run_one(
            subtree=subtree,
            dim=dim,
            model_name=setting["model"],
            optimizer_name=setting["optimizer"],
            seed=seed,
            epochs=cfg["experiment"]["epochs"],
            batch_size=cfg["experiment"]["batch_size"],
            lr=cfg["experiment"]["lr"],
            weight_decay=cfg["experiment"].get("weight_decay", 0.0),
            embeddings_path=embeddings_path,
            edges_path=cfg["data"]["edges_path"],
            results_root=results_root,
            test_size=cfg["data"].get("test_size", 0.2),
            split_protocol=cfg["data"].get("split_protocol", "stratified"),
            device=cfg["experiment"].get("device", "cpu"),
        )
    if not args.skip_finalize:
        aggregate_results(results_root)
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["visualize", "--config", str(config_path)]
            visualize_main()
        finally:
            sys.argv = old_argv


if __name__ == "__main__":
    main()
