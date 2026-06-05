from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    y_pred = (logits >= 0).astype(np.float32)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


@torch.no_grad()
def evaluate_model(model, x: torch.Tensor, y: torch.Tensor, loss_fn) -> dict[str, float]:
    model.eval()
    logits = model(x)
    loss = float(loss_fn(logits, y).item())
    metrics = binary_metrics(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
    return {"loss": loss, **metrics}


def aggregate_results(results_root: str | Path = "results") -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(results_root)
    rows = []
    for path in sorted((root / "logs").glob("*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        final = df.iloc[-1].to_dict()
        rows.append(final)
    all_runs = pd.DataFrame(rows)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    if all_runs.empty:
        empty = pd.DataFrame()
        empty.to_csv(root / "tables" / "main_results.csv", index=False)
        empty.to_csv(root / "tables" / "optimizer_stability.csv", index=False)
        return empty, empty
    group_cols = ["subtree", "dim", "model", "optimizer"]
    main = all_runs.groupby(group_cols).agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        test_loss_mean=("test_loss", "mean"),
        test_loss_std=("test_loss", "std"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        test_precision_mean=("test_precision", "mean"),
        test_precision_std=("test_precision", "std"),
        test_recall_mean=("test_recall", "mean"),
        test_recall_std=("test_recall", "std"),
        test_f1_mean=("test_f1", "mean"),
        test_f1_std=("test_f1", "std"),
        grad_norm_mean=("grad_norm", "mean"),
        grad_norm_std=("grad_norm", "std"),
        avg_param_norm_mean=("avg_param_norm", "mean"),
        avg_param_norm_std=("avg_param_norm", "std"),
        max_param_norm_mean=("max_param_norm", "mean"),
        max_param_norm_std=("max_param_norm", "std"),
        projection_count_mean=("projection_count", "mean"),
        projection_count_std=("projection_count", "std"),
        clipping_count_mean=("clipping_count", "mean"),
        clipping_count_std=("clipping_count", "std"),
        runtime_sec_mean=("runtime_sec", "mean"),
        runtime_sec_std=("runtime_sec", "std"),
    ).reset_index()
    stability = all_runs.groupby(["subtree", "dim", "optimizer"]).agg(
        seed_variance=("test_f1", "var"),
        mean_projection_count=("projection_count", "mean"),
        std_projection_count=("projection_count", "std"),
        mean_clipping_count=("clipping_count", "mean"),
        mean_grad_norm=("grad_norm", "mean"),
        runtime_per_epoch=("runtime_sec", "mean"),
    ).reset_index()
    main.to_csv(root / "tables" / "main_results.csv", index=False)
    stability.to_csv(root / "tables" / "optimizer_stability.csv", index=False)
    (root / "tables" / "main_results_latex.txt").write_text(main.to_latex(index=False), encoding="utf-8")
    return main, stability
