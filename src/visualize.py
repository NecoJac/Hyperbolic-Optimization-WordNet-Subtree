from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .data_wordnet import ensure_synthetic_data, load_embeddings, load_wordnet_edges, make_subtree_dataset
from .models import EuclideanLogisticRegression, HyperbolicMLR, LogMapEuclideanLogisticRegression
from .poincare_ops import project_to_ball


MODEL_LABELS = {
    "euclidean_lr": "Euclidean LR",
    "logmap_lr": "Log-map LR",
    "hyperbolic_mlr": "Hyperbolic MLR",
}


def _logs(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted((root / "logs").glob("*.csv")) if "trajectory" not in path.name]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_curves(root: Path) -> None:
    df = _logs(root)
    if df.empty:
        return
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for (subtree, dim), sub in df.groupby(["subtree", "dim"]):
        short = subtree.split(".")[0]
        for metric, name, ylabel in [("train_loss", "loss_curve", "Train loss"), ("test_f1", "f1_curve", "Test F1")]:
            plt.figure(figsize=(7, 4.2))
            for key, run in sub.groupby(["model", "optimizer"]):
                curve = run.groupby("epoch")[metric].mean()
                plt.plot(curve.index, curve.values, label=f"{MODEL_LABELS.get(key[0], key[0])} + {key[1]}")
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(fig_dir / f"{name}_{short}_dim{dim}.png", dpi=180)
            plt.close()
        plt.figure(figsize=(7, 4.2))
        for key, run in sub.groupby(["model", "optimizer"]):
            curve = run.groupby("epoch")[["avg_param_norm", "test_f1"]].mean()
            plt.plot(curve.index, curve["avg_param_norm"], label=f"{key[1]} norm")
            plt.plot(curve.index, curve["test_f1"], linestyle="--", label=f"{key[1]} F1")
        plt.xlabel("Epoch")
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"norm_f1_{short}_dim{dim}.png", dpi=180)
        plt.close()


def plot_main_bar(root: Path) -> None:
    table = root / "tables" / "main_results.csv"
    if not table.exists():
        return
    df = pd.read_csv(table)
    if df.empty:
        return
    labels = [f"{r.subtree.split('.')[0]}\n{r.dim}d\n{r.optimizer}" for r in df.itertuples()]
    plt.figure(figsize=(max(9, len(df) * 0.38), 4.8))
    plt.bar(np.arange(len(df)), df["test_f1_mean"], yerr=df["test_f1_std"].fillna(0), color="#4c78a8")
    plt.xticks(np.arange(len(df)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Test F1")
    plt.tight_layout()
    plt.savefig(root / "figures" / "main_f1_barplot.png", dpi=180)
    plt.close()


def _load_checkpoint_model(path: Path, model_name: str, dim: int):
    if model_name == "euclidean_lr":
        model = EuclideanLogisticRegression(dim)
    elif model_name == "logmap_lr":
        model = LogMapEuclideanLogisticRegression(dim)
    else:
        model = HyperbolicMLR(dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _parse_stem(stem: str, subtree: str) -> tuple[str, str]:
    prefix = f"{subtree}_dim2_"
    suffix = "_seed0"
    middle = stem[len(prefix):-len(suffix)]
    for model_name in ["hyperbolic_mlr", "euclidean_lr", "logmap_lr"]:
        marker = f"{model_name}_"
        if middle.startswith(marker):
            return model_name, middle[len(marker):]
    raise ValueError(f"Cannot parse checkpoint name: {stem}")


def _resolve_embeddings_path(embeddings_path: str | dict, dim: int) -> str:
    if isinstance(embeddings_path, dict):
        return str(embeddings_path[str(dim)])
    return str(embeddings_path)


def plot_boundary(root: Path, subtree: str, embeddings_path: str | dict, edges_path: str) -> None:
    embeddings_path = _resolve_embeddings_path(embeddings_path, 2)
    ensure_synthetic_data(embeddings_path, edges_path)
    embeddings = load_embeddings(embeddings_path, dim=2)
    edges = load_wordnet_edges(edges_path)
    _, x_np, y_np = make_subtree_dataset(subtree, embeddings, edges)
    x = torch.tensor(x_np, dtype=torch.float32)
    xx, yy = np.meshgrid(np.linspace(-0.98, 0.98, 160), np.linspace(-0.98, 0.98, 160))
    grid = np.c_[xx.ravel(), yy.ravel()]
    mask = np.linalg.norm(grid, axis=1) < 0.98
    grid_t = torch.tensor(grid[mask], dtype=torch.float32)
    ckpts = sorted((root / "checkpoints").glob(f"{subtree}_dim2_*_seed0.pt"))
    if not ckpts:
        return
    selected = ckpts[:5]
    fig, axes = plt.subplots(1, len(selected), figsize=(3.1 * len(selected), 3.2), squeeze=False)
    for ax, ckpt in zip(axes[0], selected):
        model_name, optimizer = _parse_stem(ckpt.stem, subtree)
        try:
            model = _load_checkpoint_model(ckpt, model_name, 2)
        except Exception:
            continue
        logits = np.full(len(grid), np.nan)
        with torch.no_grad():
            logits[mask] = model(project_to_ball(grid_t)).numpy()
        z = logits.reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=[-100, 0, 100], alpha=0.15, colors=["#e45756", "#54a24b"])
        ax.contour(xx, yy, z, levels=[0], colors="black", linewidths=1)
        ax.scatter(x_np[y_np == 0, 0], x_np[y_np == 0, 1], s=7, alpha=0.35, label="negative")
        ax.scatter(x_np[y_np == 1, 0], x_np[y_np == 1, 1], s=8, alpha=0.65, label="positive")
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="black", linewidth=1))
        ax.set_aspect("equal")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.axis("off")
        ax.set_title(ckpt.stem.replace(f"{subtree}_dim2_", "").replace("_seed0", ""), fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "figures" / f"boundary_{subtree.split('.')[0]}_dim2.png", dpi=180)
    plt.close()


def plot_trajectory(root: Path, subtree: str) -> None:
    paths = sorted((root / "logs").glob(f"{subtree}_dim2_hyperbolic_mlr_*_seed0_trajectory.csv"))
    if not paths:
        return
    plt.figure(figsize=(4.5, 4.5))
    for path in paths:
        df = pd.read_csv(path)
        if {"x0", "x1"}.issubset(df.columns):
            label = path.stem.replace(f"{subtree}_dim2_hyperbolic_mlr_", "").replace("_seed0_trajectory", "")
            plt.plot(df["x0"], df["x1"], marker="o", markersize=2, linewidth=1, label=label)
    plt.gca().add_patch(plt.Circle((0, 0), 1, fill=False, color="black", linewidth=1))
    plt.gca().set_aspect("equal")
    plt.xlim(-1.02, 1.02)
    plt.ylim(-1.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "figures" / f"trajectory_{subtree.split('.')[0]}_dim2.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/small_experiment.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["outputs"]["root"])
    (root / "figures").mkdir(parents=True, exist_ok=True)
    plot_curves(root)
    plot_main_bar(root)
    for subtree in cfg["outputs"].get("figures_subtrees", cfg["experiment"]["subtrees"]):
        plot_boundary(root, subtree, cfg["data"].get("embeddings_by_dim", cfg["data"].get("embeddings_path")), cfg["data"]["edges_path"])
        plot_trajectory(root, subtree)


if __name__ == "__main__":
    main()
