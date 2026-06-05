#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _torch_load(path: Path) -> dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _lorentz_to_poincare(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[1] < 2:
        raise ValueError("Lorentz embeddings must have at least time + one spatial dimension")
    time = embeddings[:, :1]
    spatial = embeddings[:, 1:]
    return spatial / np.clip(time + 1.0, 1e-12, None)


def load_checkpoint_embeddings(checkpoint_path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    state = _torch_load(checkpoint_path)
    if "objects" not in state:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain an 'objects' list")
    if "embeddings" in state:
        embeddings = _as_numpy(state["embeddings"])
    elif "model" in state and "lt.weight" in state["model"]:
        embeddings = _as_numpy(state["model"]["lt.weight"])
    else:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain embeddings or model['lt.weight']")

    objects = [str(obj) for obj in state["objects"]]
    conf = state.get("conf", {}) or {}
    manifold = str(conf.get("manifold", "poincare")).lower()
    if manifold == "lorentz":
        embeddings = _lorentz_to_poincare(embeddings)
    elif manifold != "poincare":
        raise ValueError(f"Unsupported checkpoint manifold: {manifold}")

    embeddings = embeddings.astype(np.float64, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    max_norm = 1.0 - 1e-5
    scale = np.minimum(1.0, max_norm / np.clip(norms, 1e-15, None))
    embeddings = embeddings * scale
    if len(objects) != embeddings.shape[0]:
        raise ValueError(f"objects length {len(objects)} != embedding rows {embeddings.shape[0]}")
    return objects, embeddings, conf


def convert_edges(closure_path: Path, edges_out: Path) -> None:
    df = pd.read_csv(closure_path)
    required = {"id1", "id2"}
    if not required.issubset(df.columns):
        raise ValueError(f"{closure_path} must contain columns {sorted(required)}")
    # facebookresearch/poincare-embeddings stores id1=hyponym/child, id2=hypernym/ancestor.
    edges = df[["id2", "id1"]].drop_duplicates().rename(columns={"id2": "parent", "id1": "child"})
    edges_out.parent.mkdir(parents=True, exist_ok=True)
    edges.sort_values(["parent", "child"]).to_csv(edges_out, index=False)


def convert_embeddings(checkpoint_path: Path, embeddings_out: Path) -> None:
    objects, embeddings, conf = load_checkpoint_embeddings(checkpoint_path)
    embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    with embeddings_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", *[f"x{i}" for i in range(embeddings.shape[1])]])
        for node, vector in sorted(zip(objects, embeddings), key=lambda item: item[0]):
            writer.writerow([node, *[f"{float(x):.17g}" for x in vector]])
    print(f"Exported {len(objects)} {conf.get('manifold', 'poincare')} embeddings with dim={embeddings.shape[1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert facebookresearch/poincare-embeddings WordNet outputs to this project's CSV format.")
    parser.add_argument("--closure", default="poincare-embeddings/wordnet/noun_closure.csv", help="Path to noun_closure.csv or mammal_closure.csv")
    parser.add_argument("--checkpoint", default="poincare-embeddings/nouns.bin", help="Path to nouns.bin, mammals.pth, or another checkpoint")
    parser.add_argument("--edges-out", default="data/processed/wordnet_edges.csv")
    parser.add_argument("--embeddings-out", default="data/embeddings/wordnet_embeddings.csv")
    args = parser.parse_args()

    closure = Path(args.closure)
    checkpoint = Path(args.checkpoint)
    if not closure.exists():
        raise FileNotFoundError(closure)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    convert_edges(closure, Path(args.edges_out))
    convert_embeddings(checkpoint, Path(args.embeddings_out))
    print(f"Wrote edges to {args.edges_out}")
    print(f"Wrote embeddings to {args.embeddings_out}")


if __name__ == "__main__":
    main()
