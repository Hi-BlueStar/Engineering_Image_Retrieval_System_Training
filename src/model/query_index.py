#!/usr/bin/env python
"""
Query a FAISS index built from SimSiam embeddings.

Example:
    python -m src.query_index --query_image path/to/query.jpg --index_path results/gallery.index --mapping_path results/index_to_path.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import torch

from src.model.simsiam import SimSiam, embed_images, make_norm


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search similar images using a FAISS index built on SimSiam embeddings."
    )
    parser.add_argument(
        "--query_image", required=True, type=Path, help="Path to the query image."
    )
    parser.add_argument(
        "--index_path",
        type=Path,
        default=Path("results") / "gallery.index",
        help="FAISS index file.",
    )
    parser.add_argument(
        "--mapping_path",
        type=Path,
        default=Path("results") / "index_to_path.json",
        help="JSON mapping from index to image paths.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("results") / "simsiam_model.pth",
        help="Trained SimSiam weights.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="SimSiam backbone (must match training).",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size used during training."
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of nearest neighbors to retrieve."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cpu or cuda)."
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {args.index_path}")
    if not args.mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {args.mapping_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
    if not args.query_image.exists():
        raise FileNotFoundError(f"Query image not found: {args.query_image}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    with args.mapping_path.open("r", encoding="utf-8") as f:
        index_to_path = json.load(f)

    index = faiss.read_index(str(args.index_path))
    dim = index.d

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = make_norm(mean, std)

    model = SimSiam(backbone=args.backbone).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        embedding = embed_images(
            model,
            [args.query_image],
            norm,
            device=device,
            batch=1,
            img_size=args.img_size,
        )
    if embedding.numel() == 0:
        raise RuntimeError("Failed to compute embedding for the query image.")

    if embedding.shape[1] != dim:
        raise ValueError(
            f"Embedding dimension {embedding.shape[1]} does not match index dimension {dim}."
        )

    query_vec = embedding.cpu().numpy()
    top_k = min(args.top_k, index.ntotal)
    if top_k == 0:
        raise RuntimeError("FAISS index is empty.")

    scores, indices = index.search(query_vec, top_k)
    scores = scores[0]
    indices = indices[0]

    print(f"Top-{top_k} similar images to {args.query_image}:")
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        if idx == -1:
            continue
        path = index_to_path.get(str(idx), "<missing>")
        print(f"{rank:02d}. {path} (score={score:.4f})")


if __name__ == "__main__":
    main()
