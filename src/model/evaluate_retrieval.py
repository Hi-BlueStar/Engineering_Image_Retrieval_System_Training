#!/usr/bin/env python
"""
Evaluate retrieval metrics on a FAISS index built from SimSiam embeddings.

Example:
    python -m src.model.evaluate_retrieval \
        --query_dir data/queries \
        --model_path results/simsiam_model.pth \
        --index_path results/gallery.index \
        --mapping_path results/index_to_path.json \
        --details_json results/eval_details.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from pathlib import Path

import faiss
import torch

from src.model.simsiam import SimSiam, embed_images, make_norm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality using a SimSiam encoder and FAISS index."
    )
    parser.add_argument(
        "--query_dir",
        required=True,
        type=Path,
        help="Directory containing query images.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("results") / "simsiam_model.pth",
        help="Trained SimSiam weights.",
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
        help="Index-to-path JSON mapping.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50"],
        help="SimSiam backbone (must match training).",
    )
    parser.add_argument(
        "--img_size", type=int, default=224, help="Image size used for embedding."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for query embedding."
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of nearest neighbors to examine."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override device (cpu or cuda)."
    )
    parser.add_argument(
        "--query_labels_csv",
        type=Path,
        help="Optional CSV with 'path,label' for queries.",
    )
    parser.add_argument(
        "--gallery_labels_csv",
        type=Path,
        help="Optional CSV with 'path,label' for gallery entries.",
    )
    parser.add_argument(
        "--skip_same_path",
        action="store_true",
        help="Skip gallery results that match the exact query path.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("results") / "eval_metrics.json",
        help="Where to store aggregated metrics.",
    )
    parser.add_argument(
        "--details_json",
        type=Path,
        help="Optional per-query JSON dump of retrieval results.",
    )
    parser.add_argument(
        "--results_csv", type=Path, help="Optional per-query CSV summary."
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(
            f"No images with extensions {sorted(IMAGE_EXTS)} under {root}"
        )
    return files


def load_labels_csv(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        first_line = f.readline()
        f.seek(0)
        if "," in first_line and "path" in first_line and "label" in first_line:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                mapping[str(Path(row["path"]).resolve())] = row["label"]
        else:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) < 2:
                    raise ValueError(f"Invalid row in {path}: {row}")
                mapping[str(Path(row[0]).resolve())] = row[1]
    return mapping


def lookup_label(path: Path, label_map: dict[str, str]) -> str:
    resolved = str(path.resolve())
    if resolved in label_map:
        return label_map[resolved]
    as_str = str(path)
    if as_str in label_map:
        return label_map[as_str]
    return path.parent.name


def write_results_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {args.index_path}")
    if not args.mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {args.mapping_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    query_paths = list_images(args.query_dir)
    if len(query_paths) == 0:
        raise RuntimeError("No query images found.")

    with args.mapping_path.open("r", encoding="utf-8") as f:
        index_to_path_raw = json.load(f)
    index_to_path = {int(k): Path(v) for k, v in index_to_path_raw.items()}

    query_labels = load_labels_csv(args.query_labels_csv)
    gallery_labels = load_labels_csv(args.gallery_labels_csv)

    model = SimSiam(backbone=args.backbone).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = make_norm(mean, std)

    embed_start = time.perf_counter()
    queries_tensor = embed_images(
        model,
        query_paths,
        norm,
        device=device,
        batch=args.batch_size,
        img_size=args.img_size,
    )
    embed_time = time.perf_counter() - embed_start
    if queries_tensor.numel() == 0:
        raise RuntimeError("Failed to compute query embeddings.")

    index = faiss.read_index(str(args.index_path))
    dim = index.d
    if queries_tensor.shape[1] != dim:
        raise ValueError(
            f"Embedding dimension {queries_tensor.shape[1]} does not match index dimension {dim}."
        )

    queries_np = queries_tensor.cpu().numpy()
    top_k = min(args.top_k, index.ntotal)
    if top_k == 0:
        raise RuntimeError("FAISS index is empty.")

    search_start = time.perf_counter()
    scores, indices = index.search(queries_np, top_k)
    search_time = time.perf_counter() - search_start

    evaluated = len(query_paths)
    top1_hits = 0
    topk_hits = 0
    precision_sum = 0.0
    top1_score_sum = 0.0
    zero_result_queries = 0
    per_query_rows: list[dict[str, object]] = []
    per_query_details: list[dict[str, object]] = []

    for idx_query, query_path in enumerate(query_paths):
        query_label = lookup_label(query_path, query_labels)
        row_indices = indices[idx_query]
        row_scores = scores[idx_query]

        retrieved = []
        for idx_candidate, score in zip(row_indices, row_scores):
            if idx_candidate == -1:
                continue
            candidate_path = index_to_path.get(int(idx_candidate))
            if candidate_path is None:
                continue
            if args.skip_same_path and candidate_path.resolve() == query_path.resolve():
                continue
            retrieved.append((candidate_path, float(score)))
            if len(retrieved) >= top_k:
                break

        if not retrieved:
            zero_result_queries += 1
            per_query_rows.append(
                {
                    "query_path": str(query_path),
                    "query_label": query_label,
                    "top1_path": "",
                    "top1_score": "",
                    "top1_match": 0,
                    "matches_in_topk": 0,
                    "precision_at_k": 0.0,
                }
            )
            per_query_details.append(
                {
                    "query_path": str(query_path),
                    "query_label": query_label,
                    "retrieved": [],
                    "top1_match": False,
                    "precision_at_k": 0.0,
                }
            )
            continue

        top1_path, top1_score = retrieved[0]
        top1_label = lookup_label(top1_path, gallery_labels)
        top1_match = int(top1_label == query_label)
        top1_hits += top1_match
        top1_score_sum += top1_score

        matches = 0
        retrieved_records = []
        for rank, (path, score) in enumerate(retrieved, start=1):
            label = lookup_label(path, gallery_labels)
            match = int(label == query_label)
            matches += match
            retrieved_records.append(
                {
                    "rank": rank,
                    "path": str(path),
                    "score": score,
                    "label": label,
                    "match": match,
                }
            )
        precision_at_k = matches / min(top_k, len(retrieved))
        precision_sum += precision_at_k
        topk_hits += int(matches > 0)

        per_query_rows.append(
            {
                "query_path": str(query_path),
                "query_label": query_label,
                "top1_path": str(top1_path),
                "top1_score": top1_score,
                "top1_match": top1_match,
                "matches_in_topk": matches,
                "precision_at_k": precision_at_k,
            }
        )
        per_query_details.append(
            {
                "query_path": str(query_path),
                "query_label": query_label,
                "top1_match": bool(top1_match),
                "precision_at_k": precision_at_k,
                "retrieved": retrieved_records,
            }
        )

    top1_accuracy = top1_hits / evaluated if evaluated else 0.0
    hit_rate = topk_hits / evaluated if evaluated else 0.0
    mean_precision = precision_sum / evaluated if evaluated else 0.0
    mean_top1_score = (
        (top1_score_sum / (evaluated - zero_result_queries))
        if (evaluated - zero_result_queries) > 0
        else 0.0
    )
    avg_query_time_ms = (
        ((embed_time + search_time) / evaluated * 1000) if evaluated else 0.0
    )

    metrics = {
        "total_queries": evaluated,
        "queries_without_results": zero_result_queries,
        "top1_accuracy": top1_accuracy,
        "hit_rate_at_k": hit_rate,
        "mean_precision_at_k": mean_precision,
        "mean_top1_score": mean_top1_score,
        "embedding_time_seconds": embed_time,
        "search_time_seconds": search_time,
        "avg_query_time_ms": avg_query_time_ms,
        "device": device,
        "top_k": top_k,
        "skip_same_path": args.skip_same_path,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if args.details_json:
        args.details_json.parent.mkdir(parents=True, exist_ok=True)
        with args.details_json.open("w", encoding="utf-8") as f:
            json.dump(per_query_details, f, indent=2)

    if args.results_csv:
        args.results_csv.parent.mkdir(parents=True, exist_ok=True)
        write_results_csv(per_query_rows, args.results_csv)

    print(
        f"Evaluated {evaluated} queries. Top-1 accuracy={top1_accuracy:.3f}, Hit@{top_k}={hit_rate:.3f}, mP@{top_k}={mean_precision:.3f}"
    )
    print(f"Metrics written to {args.output_path}")


if __name__ == "__main__":
    main()
