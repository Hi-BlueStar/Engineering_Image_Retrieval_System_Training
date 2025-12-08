#!/usr/bin/env python
"""
Embed gallery images with a trained SimSiam model and build a FAISS index.

Example:
    python -m src.build_index --gallery_dir data/gallery --model_path results/simsiam_model.pth
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import faiss
import torch

from src.model.simsiam import SimSiam, embed_images, make_norm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(
            f"No images with extensions {sorted(IMAGE_EXTS)} under {root}"
        )
    return files


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS index for image gallery.")
    parser.add_argument(
        "--gallery_dir", required=True, type=Path, help="Directory with gallery images."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("results") / "simsiam_model.pth",
        help="Path to trained SimSiam weights.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory to store index files.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="SimSiam backbone (must match training).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for embedding."
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size used during training."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cpu or cuda)."
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gallery_paths = list_images(args.gallery_dir)
    print(f"Found {len(gallery_paths)} gallery images under {args.gallery_dir}")

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = make_norm(mean, std)

    model = SimSiam(backbone=args.backbone).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        embeddings = embed_images(
            model,
            gallery_paths,
            norm,
            device=device,
            batch=args.batch_size,
            img_size=args.img_size,
        )
    if embeddings.numel() == 0:
        raise RuntimeError(
            "No embeddings were generated. Check gallery directory and preprocessing."
        )

    if embeddings.dtype != torch.float32:
        embeddings = embeddings.float()
    vecs = embeddings.cpu().numpy()

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss_path = output_dir / "gallery.index"
    faiss.write_index(index, str(faiss_path))

    mapping = {str(i): str(path) for i, path in enumerate(gallery_paths)}
    mapping_path = output_dir / "index_to_path.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index to {faiss_path}")
    print(f"Saved index-to-path mapping to {mapping_path}")


if __name__ == "__main__":
    main()

# python -m src.build_index --gallery_dir … --model_path results/simsiam_model.pth
