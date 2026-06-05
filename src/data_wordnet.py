from __future__ import annotations

from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOTS = ["ENTITY.N.01", "ANIMAL.N.01", "MAMMAL.N.01", "GROUP.N.01", "WORKER.N.01"]


def _synthetic_tree() -> tuple[list[tuple[str, str]], dict[str, tuple[float, float]]]:
    edges: list[tuple[str, str]] = []
    coords: dict[str, tuple[float, float]] = {"ENTITY.N.01": (0.0, 0.0)}
    branches = {
        "ANIMAL.N.01": (0.0, 26),
        "GROUP.N.01": (2.2, 24),
        "WORKER.N.01": (4.2, 22),
        "ARTIFACT.N.01": (5.4, 26),
    }
    for root, (angle, n_children) in branches.items():
        edges.append(("ENTITY.N.01", root))
        coords[root] = (0.28, angle)
        for i in range(n_children):
            child = f"{root.split('.')[0]}_{i:02d}.N.01"
            child_angle = angle + (i - n_children / 2) * 0.035
            child_radius = 0.55 + 0.16 * ((i % 5) / 4)
            edges.append((root, child))
            coords[child] = (child_radius, child_angle)
            for j in range(2):
                leaf = f"{child.replace('.N.01', '')}_{j}.N.01"
                leaf_angle = child_angle + (j - 0.5) * 0.018
                leaf_radius = 0.76 + 0.08 * ((i + j) % 3)
                edges.append((child, leaf))
                coords[leaf] = (leaf_radius, leaf_angle)
    edges.append(("ANIMAL.N.01", "MAMMAL.N.01"))
    coords["MAMMAL.N.01"] = (0.43, 0.12)
    for i in range(34):
        child = f"MAMMAL_{i:02d}.N.01"
        angle = 0.12 + (i - 17) * 0.026
        radius = 0.63 + 0.18 * ((i % 7) / 6)
        edges.append(("MAMMAL.N.01", child))
        coords[child] = (radius, angle)
        for j in range(2):
            leaf = f"MAMMAL_{i:02d}_{j}.N.01"
            edges.append((child, leaf))
            coords[leaf] = (0.82 + 0.05 * j, angle + (j - 0.5) * 0.014)
    return edges, coords


def ensure_synthetic_data(embeddings_path: str | Path, edges_path: str | Path, max_dim: int = 10) -> None:
    embeddings_path = Path(embeddings_path)
    edges_path = Path(edges_path)
    if embeddings_path.exists() and edges_path.exists():
        return
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    edges, polar = _synthetic_tree()
    rng = np.random.default_rng(7)
    rows = []
    for node, (radius, angle) in polar.items():
        base = np.zeros(max_dim, dtype=np.float64)
        base[0] = radius * np.cos(angle)
        base[1] = radius * np.sin(angle)
        if max_dim > 2:
            base[2:] = rng.normal(0.0, 0.015, size=max_dim - 2)
        norm = np.linalg.norm(base)
        if norm >= 0.98:
            base *= 0.98 / norm
        rows.append({"node": node, **{f"x{i}": base[i] for i in range(max_dim)}})
    pd.DataFrame(edges, columns=["parent", "child"]).to_csv(edges_path, index=False)
    pd.DataFrame(rows).sort_values("node").to_csv(embeddings_path, index=False)


def load_embeddings(path: str | Path, dim: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("x")]
    cols = sorted(cols, key=lambda c: int(c[1:]))
    if dim is not None:
        cols = cols[:dim]
    return df[["node", *cols]]


def load_wordnet_edges(path: str | Path) -> list[tuple[str, str]]:
    df = pd.read_csv(path)
    return list(df[["parent", "child"]].itertuples(index=False, name=None))


def get_subtree_nodes(root: str, edges: Iterable[tuple[str, str]]) -> set[str]:
    graph = nx.DiGraph()
    normalized_edges = [(str(parent), str(child)) for parent, child in edges]
    graph.add_edges_from(normalized_edges)
    if root not in graph and root.lower() in graph:
        root = root.lower()
    if root not in graph and root.upper() in graph:
        root = root.upper()
    if root not in graph:
        raise ValueError(f"Unknown subtree root: {root}")
    return {root, *nx.descendants(graph, root)}


def make_subtree_dataset(root: str, embeddings: pd.DataFrame, edges: Iterable[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    subtree_nodes = get_subtree_nodes(root, edges)
    feature_cols = [c for c in embeddings.columns if c.startswith("x")]
    nodes = embeddings["node"].to_numpy()
    x = embeddings[feature_cols].to_numpy(dtype=np.float32)
    y = np.array([node in subtree_nodes for node in nodes], dtype=np.float32)
    return nodes, x, y


def train_test_split_subtree(
    labels: np.ndarray,
    seed: int,
    test_size: float = 0.2,
    protocol: str = "stratified",
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(labels))
    if protocol == "stratified":
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=labels)
        return train_idx, test_idx
    if protocol == "subtree_balanced":
        rng = np.random.default_rng(seed)
        train_parts = []
        test_parts = []
        for value in [0.0, 1.0]:
            class_idx = idx[labels == value].copy()
            rng.shuffle(class_idx)
            n_test = int(round(len(class_idx) * test_size))
            n_test = min(max(n_test, 1), len(class_idx) - 1)
            test_parts.append(class_idx[:n_test])
            train_parts.append(class_idx[n_test:])
        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        return train_idx, test_idx
    raise ValueError(f"Unsupported split protocol: {protocol}")
