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
    "euclidean_lr": "E-LR",
    "logmap_lr": "LogMap-LR",
    "hyperbolic_mlr": "H-MLR",
}

OPTIMIZER_LABELS = {
    "adam": "Adam",
    "sgd": "SGD",
    "projected_sgd": "Proj. SGD",
    "projected_adam": "Proj. Adam",
    "rsgd": "RSGD",
    "radam": "RAdam",
}

HYPERBOLIC_MLR_OPTIMIZERS = ("radam", "projected_adam", "rsgd", "projected_sgd")
PAPER_LINE_FIGSIZE = (5.4, 3.35)
PAPER_TRAJECTORY_FIGSIZE = (3.45, 3.35)


def setting_label(model_name: str, optimizer_name: str) -> str:
    return f"{MODEL_LABELS.get(model_name, model_name)} + {OPTIMIZER_LABELS.get(optimizer_name, optimizer_name)}"


def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.45,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _logs(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted((root / "logs").glob("*.csv")) if "trajectory" not in path.name]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_curves(root: Path) -> None:
    apply_paper_style()
    df = _logs(root)
    if df.empty:
        return
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for (subtree, dim), sub in df.groupby(["subtree", "dim"]):
        short = subtree.split(".")[0]
        for metric, name, ylabel in [("train_loss", "loss_curve", "Train loss"), ("test_f1", "f1_curve", "Test F1")]:
            plt.figure(figsize=PAPER_LINE_FIGSIZE)
            plot_df = sub
            if name == "f1_curve":
                plot_df = sub[(sub["model"] == "hyperbolic_mlr") & sub["optimizer"].isin(HYPERBOLIC_MLR_OPTIMIZERS)]
            if plot_df.empty:
                plt.close()
                continue
            if name == "f1_curve":
                groups = [
                    (("hyperbolic_mlr", opt), plot_df[plot_df["optimizer"] == opt])
                    for opt in HYPERBOLIC_MLR_OPTIMIZERS
                ]
            else:
                groups = list(plot_df.groupby(["model", "optimizer"]))
            for key, run in groups:
                if run.empty:
                    continue
                curve = run.groupby("epoch")[metric].mean()
                label = OPTIMIZER_LABELS.get(key[1], key[1]) if name == "f1_curve" else setting_label(key[0], key[1])
                plt.plot(curve.index, curve.values, label=label)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.22, linewidth=0.6)
            plt.legend(frameon=False, loc="best")
            plt.tight_layout()
            plt.savefig(fig_dir / f"{name}_{short}_dim{dim}.png", dpi=300)
            plt.close()
        plt.figure(figsize=PAPER_LINE_FIGSIZE)
        for key, run in sub.groupby(["model", "optimizer"]):
            curve = run.groupby("epoch")[["avg_param_norm", "test_f1"]].mean()
            label = setting_label(key[0], key[1])
            plt.plot(curve.index, curve["avg_param_norm"], label=f"{label} norm")
            plt.plot(curve.index, curve["test_f1"], linestyle="--", label=f"{label} F1")
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.22, linewidth=0.6)
        plt.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"norm_f1_{short}_dim{dim}.png", dpi=300)
        plt.close()


def plot_main_bar(root: Path) -> None:
    apply_paper_style()
    table = root / "tables" / "main_results.csv"
    if not table.exists():
        return
    df = pd.read_csv(table)
    if df.empty:
        return
    labels = [f"{r.subtree.split('.')[0]}\n{r.dim}d\n{setting_label(r.model, r.optimizer)}" for r in df.itertuples()]
    plt.figure(figsize=(max(11, len(df) * 0.46), 5.4))
    plt.bar(np.arange(len(df)), df["test_f1_mean"], yerr=df["test_f1_std"].fillna(0), color="#4c78a8")
    plt.xticks(np.arange(len(df)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Test F1")
    plt.tight_layout()
    plt.savefig(root / "figures" / "main_f1_barplot.png", dpi=300)
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
    apply_paper_style()
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
    selected = [
        path for path in ckpts
        if _parse_stem(path.stem, subtree)[0] == "hyperbolic_mlr"
        and _parse_stem(path.stem, subtree)[1] in HYPERBOLIC_MLR_OPTIMIZERS
    ]
    selected = sorted(selected, key=lambda path: HYPERBOLIC_MLR_OPTIMIZERS.index(_parse_stem(path.stem, subtree)[1]))
    if not selected:
        return
    fig, axes = plt.subplots(1, len(selected), figsize=(2.25 * len(selected), 2.35), squeeze=False)
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
        ax.set_title(setting_label(model_name, optimizer), fontsize=8)
    plt.tight_layout()
    plt.savefig(root / "figures" / f"boundary_{subtree.split('.')[0]}_dim2.png", dpi=300)
    plt.close()


def plot_trajectory(root: Path, subtree: str) -> None:
    apply_paper_style()
    paths_by_optimizer = {
        path.stem.replace(f"{subtree}_dim2_hyperbolic_mlr_", "").replace("_seed0_trajectory", ""): path
        for path in (root / "logs").glob(f"{subtree}_dim2_hyperbolic_mlr_*_seed0_trajectory.csv")
    }
    paths = [paths_by_optimizer[opt] for opt in HYPERBOLIC_MLR_OPTIMIZERS if opt in paths_by_optimizer]
    if not paths:
        return
    plt.figure(figsize=PAPER_TRAJECTORY_FIGSIZE)
    for path in paths:
        df = pd.read_csv(path)
        if {"x0", "x1"}.issubset(df.columns):
            optimizer = path.stem.replace(f"{subtree}_dim2_hyperbolic_mlr_", "").replace("_seed0_trajectory", "")
            plt.plot(df["x0"], df["x1"], marker="o", markersize=1.9, linewidth=1.15, label=OPTIMIZER_LABELS.get(optimizer, optimizer))
    plt.gca().add_patch(plt.Circle((0, 0), 1, fill=False, color="black", linewidth=1))
    plt.gca().set_aspect("equal")
    plt.xlim(-1.02, 1.02)
    plt.ylim(-1.02, 1.02)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(root / "figures" / f"trajectory_{subtree.split('.')[0]}_dim2.png", dpi=300)
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
