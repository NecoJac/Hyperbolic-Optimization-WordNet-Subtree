from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data_wordnet import (
    ensure_synthetic_data,
    load_embeddings,
    load_wordnet_edges,
    make_subtree_dataset,
    train_test_split_subtree,
)
from .evaluate import evaluate_model
from .models import EuclideanLogisticRegression, HyperbolicMLR, LogMapEuclideanLogisticRegression
from .optimizers import make_optimizer
from .poincare_ops import count_outside_ball, project_to_ball


def build_model(model_name: str, dim: int):
    if model_name == "euclidean_lr":
        return EuclideanLogisticRegression(dim)
    if model_name == "logmap_lr":
        return LogMapEuclideanLogisticRegression(dim)
    if model_name == "hyperbolic_mlr":
        return HyperbolicMLR(dim)
    raise ValueError(f"Unknown model: {model_name}")


def grad_norm(model) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm().item() ** 2)
    return total ** 0.5


def param_norms(model) -> tuple[float, float]:
    tensors = model.norm_tensors() if hasattr(model, "norm_tensors") else list(model.parameters())
    norms = torch.cat([p.detach().reshape(-1, p.shape[-1] if p.ndim > 1 else 1).norm(dim=-1).cpu() for p in tensors])
    return float(norms.mean().item()), float(norms.max().item())


_DATASET_CACHE: dict[tuple[str, str, str, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def resolve_embeddings_path(embeddings_path: str | dict, dim: int) -> str:
    if isinstance(embeddings_path, dict):
        key = str(dim)
        if key not in embeddings_path:
            raise KeyError(f"No embeddings path configured for dim={dim}")
        return str(embeddings_path[key])
    return str(embeddings_path)


def load_prepared_subtree_dataset(
    subtree: str,
    dim: int,
    embeddings_path: str | dict,
    edges_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings_path = resolve_embeddings_path(embeddings_path, dim)
    cache_key = (subtree, embeddings_path, edges_path, dim)
    if cache_key not in _DATASET_CACHE:
        ensure_synthetic_data(embeddings_path, edges_path, max_dim=max(10, dim))
        embeddings = load_embeddings(embeddings_path, dim=dim)
        edges = load_wordnet_edges(edges_path)
        nodes, x_np, y_np = make_subtree_dataset(subtree, embeddings, edges)
        x_np = project_to_ball(torch.tensor(x_np)).numpy()
        _DATASET_CACHE[cache_key] = (nodes, x_np, y_np)
    return _DATASET_CACHE[cache_key]


def run_one(
    subtree: str,
    dim: int,
    model_name: str,
    optimizer_name: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    embeddings_path: str | dict,
    edges_path: str,
    results_root: str,
    test_size: float = 0.2,
    split_protocol: str = "stratified",
    device: str = "cpu",
) -> Path:
    torch.manual_seed(seed)
    np.random.seed(seed)
    nodes, x_np, y_np = load_prepared_subtree_dataset(subtree, dim, embeddings_path, edges_path)
    train_idx, test_idx = train_test_split_subtree(y_np, seed=seed, test_size=test_size, protocol=split_protocol)

    x_train = torch.tensor(x_np[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(y_np[train_idx], dtype=torch.float32, device=device)
    x_test = torch.tensor(x_np[test_idx], dtype=torch.float32, device=device)
    y_test = torch.tensor(y_np[test_idx], dtype=torch.float32, device=device)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    model = build_model(model_name, dim).to(device)
    optimizer = make_optimizer(model, optimizer_name, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    logs_dir = Path(results_root) / "logs"
    ckpt_dir = Path(results_root) / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{subtree}_dim{dim}_{model_name}_{optimizer_name}_seed{seed}"
    log_path = logs_dir / f"{stem}.csv"
    trajectory_path = logs_dir / f"{stem}_trajectory.csv"
    fieldnames = [
        "epoch", "subtree", "dim", "model", "optimizer", "seed", "split_protocol", "train_loss", "test_loss",
        "test_accuracy", "test_precision", "test_recall", "test_f1", "grad_norm",
        "avg_param_norm", "max_param_norm", "projection_count", "clipping_count", "runtime_sec",
    ]
    trajectory_rows = []
    projection_total = 0
    clipping_total = 0
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            start = time.perf_counter()
            model.train()
            running_loss = 0.0
            running_grad_norm = 0.0
            batches = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                running_grad_norm += grad_norm(model)
                projection_total += optimizer.step(model)
                running_loss += float(loss.item()) * len(xb)
                batches += 1
            manifold_tensors = model.manifold_tensors() if hasattr(model, "manifold_tensors") else []
            clipping_total += sum(count_outside_ball(p.detach()) for p in manifold_tensors)
            train_loss = running_loss / len(train_loader.dataset)
            metrics = evaluate_model(model, x_test, y_test, loss_fn)
            avg_norm, max_norm = param_norms(model)
            runtime = time.perf_counter() - start
            writer.writerow({
                "epoch": epoch,
                "subtree": subtree,
                "dim": dim,
                "model": model_name,
                "optimizer": optimizer_name,
                "seed": seed,
                "split_protocol": split_protocol,
                "train_loss": train_loss,
                "test_loss": metrics["loss"],
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "grad_norm": running_grad_norm / max(1, batches),
                "avg_param_norm": avg_norm,
                "max_param_norm": max_norm,
                "projection_count": projection_total,
                "clipping_count": clipping_total,
                "runtime_sec": runtime,
            })
            if hasattr(model, "trajectory_point"):
                p = model.trajectory_point().numpy()
                trajectory_rows.append({"epoch": epoch, **{f"x{i}": p[i] for i in range(len(p))}})
    if trajectory_rows:
        with trajectory_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trajectory_rows)
    torch.save({"model_state": model.state_dict(), "nodes": nodes.tolist(), "test_idx": test_idx.tolist()}, ckpt_dir / f"{stem}.pt")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtree", required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--embeddings-path", default="data/embeddings/wordnet_embeddings.csv")
    parser.add_argument("--edges-path", default="data/processed/wordnet_edges.csv")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-protocol", default="stratified", choices=["stratified", "subtree_balanced"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_one(
        subtree=args.subtree,
        dim=args.dim,
        model_name=args.model,
        optimizer_name=args.optimizer,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embeddings_path=args.embeddings_path,
        edges_path=args.edges_path,
        results_root=args.results_root,
        test_size=args.test_size,
        split_protocol=args.split_protocol,
        device=args.device,
    )


if __name__ == "__main__":
    main()
