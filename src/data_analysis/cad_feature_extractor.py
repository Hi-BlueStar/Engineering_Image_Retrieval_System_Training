from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer, models
from sklearn.manifold import TSNE


logger = logging.getLogger(__name__)


def _ensure_directory(path: Path) -> None:
    """Create directory if it does not exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class PDFToImageConverter:
    """Convert PDF pages to cached images.

    Attributes:
        image_cache_dir: Root directory for cached images.
        dpi: Render resolution for PDF pages.
    """

    image_cache_dir: Path
    dpi: int = 300

    def convert_pdf(self, pdf_path: Path, label: str) -> list[Path]:
        """Convert a PDF into PNG images, using cache when available.

        Args:
            pdf_path: Path to the PDF file.
            label: Label name derived from the parent directory.

        Returns:
            A list of generated image paths (one per page). Cached files are reused.
        """
        destination_dir = self.image_cache_dir / label / pdf_path.stem
        _ensure_directory(destination_dir)
        try:
            document = fitz.open(pdf_path)
        except (RuntimeError, ValueError) as exc:
            logger.warning("Failed to open PDF %s: %s", pdf_path, exc)
            return []

        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        image_paths: list[Path] = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            output_path = destination_dir / f"page_{page_index + 1:03d}.png"
            if output_path.exists():
                image_paths.append(output_path)
                continue

            try:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(output_path)
                image_paths.append(output_path)
            except RuntimeError as exc:  # corrupted page or rendering issue
                logger.warning(
                    "Failed to render page %s (page %d): %s", pdf_path, page_index, exc
                )
                continue

        document.close()
        return image_paths


@dataclass
class OpenCLIPExtractor:
    """Extract OpenCLIP embeddings using sentence-transformers."""

    model_name: str
    device: str = "cpu"
    batch_size: int = 8
    fallback_model_name: str = "sentence-transformers/clip-ViT-L-14"
    model_name_used: str = field(init=False)
    allow_cpu_fallback: bool = True

    def __post_init__(self) -> None:
        """Load the OpenCLIP model on the configured device."""
        self.model_name_used = self.model_name
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            return
        except AttributeError as exc:
            if "hidden_size" not in str(exc):
                raise
            logger.warning(
                "Model %s missing hidden_size in config; attempting CLIPModel fallback. Original error: %s",
                self.model_name,
                exc,
            )

        try:
            clip_module = models.CLIPModel(self.model_name)
            self.model = SentenceTransformer(modules=[clip_module])
            return
        except Exception as clip_exc:  # pragma: no cover - defensive
            logger.warning(
                "CLIPModel fallback failed for %s (%s); using fallback model %s",
                self.model_name,
                clip_exc,
                self.fallback_model_name,
            )
            self.model_name_used = self.fallback_model_name
            self.model = SentenceTransformer(
                self.fallback_model_name, device=self.device
            )

    def _load_images(self, image_paths: Sequence[Path]) -> list[Image.Image]:
        """Load images into memory with basic error handling.

        Args:
            image_paths: Image file paths to load.

        Returns:
            A list of RGB PIL images. Corrupted or missing files are skipped.
        """
        loaded: list[Image.Image] = []
        for image_path in image_paths:
            try:
                image = Image.open(image_path).convert("RGB")
                loaded.append(image)
            except (FileNotFoundError, UnidentifiedImageError) as exc:
                logger.warning("Unable to read image %s: %s", image_path, exc)
        return loaded

    def extract(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Extract normalized embeddings for a batch of images.

        Args:
            image_paths: Iterable of image file paths.

        Returns:
            A numpy array of shape (N, D) containing L2-normalized embeddings.
        """
        embeddings: list[np.ndarray] = []
        loaded_any = False

        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            batch_images = self._load_images(batch_paths)
            if not batch_images:
                continue

            loaded_any = True
            current_bs = min(self.batch_size, len(batch_images))
            while True:
                try:
                    batch_embeddings = self._encode_batch(batch_images, current_bs)
                    embeddings.append(batch_embeddings)
                    break
                except RuntimeError as exc:
                    lowered = str(exc).lower()
                    if "out of memory" not in lowered:
                        raise
                    next_bs = self._handle_cuda_oom(current_bs)
                    if next_bs is None:
                        raise
                    current_bs = next_bs

        if not loaded_any or not embeddings:
            raise ValueError("No images could be loaded for embedding extraction.")

        return np.concatenate(embeddings, axis=0)

    def _encode_batch(
        self, batch_images: list[Image.Image], batch_size: int
    ) -> np.ndarray:
        """Encode a batch of images, handling different encode signatures."""
        encode_sig = inspect.signature(self.model.encode)
        if "images" in encode_sig.parameters:
            return self.model.encode(
                images=batch_images,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        # Fallback for sentence-transformers >=5 where encode expects `sentences`
        return self.model.encode(
            sentences=batch_images,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _handle_cuda_oom(self, current_bs: int) -> int | None:
        """Handle CUDA OOM by shrinking batch size or falling back to CPU."""
        logger.warning(
            "CUDA OOM encountered; current batch size: %d. Attempting recovery.",
            current_bs,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if current_bs > 1:
            new_bs = max(1, current_bs // 2)
            logger.info("Reducing batch size to %d and retrying.", new_bs)
            return new_bs

        if self.allow_cpu_fallback and self.device.startswith("cuda"):
            logger.warning("Falling back to CPU for embedding extraction to avoid OOM.")
            self.model = self.model.to("cpu")
            self.device = "cpu"
            return 1

        logger.error("Cannot recover from OOM with current settings.")
        return None


@dataclass
class ManifoldReducer:
    """Run manifold learning algorithms on embeddings."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    random_state: int | None = None
    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 1000

    def run_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute UMAP projection.

        Args:
            embeddings: High-dimensional embeddings of shape (N, D).

        Returns:
            Array of 2D coordinates with shape (N, 2).
        """
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
        )
        return reducer.fit_transform(embeddings)

    def run_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute t-SNE projection.

        Args:
            embeddings: High-dimensional embeddings of shape (N, D).

        Returns:
            Array of 2D coordinates with shape (N, 2).
        """
        perplexity = min(self.tsne_perplexity, max(5, (len(embeddings) - 1) / 3))
        tsne_kwargs = dict(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            metric=self.metric,
            random_state=self.random_state,
        )

        tsne_sig = inspect.signature(TSNE.__init__)
        if "max_iter" in tsne_sig.parameters:
            tsne_kwargs["max_iter"] = self.tsne_max_iter
        else:
            tsne_kwargs["n_iter"] = self.tsne_max_iter

        tsne = TSNE(**tsne_kwargs)
        return tsne.fit_transform(embeddings)


class VisualizationBuilder:
    """Create interactive scatter plots for embeddings."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the visualization builder.

        Args:
            output_dir: Directory where HTML artifacts are written.
        """
        self.output_dir = output_dir
        _ensure_directory(self.output_dir)

    def _build_hover_text(self, df: pd.DataFrame) -> list[str]:
        """Compose hover text strings for Plotly.

        Args:
            df: DataFrame containing file metadata.

        Returns:
            A list of HTML-formatted hover strings.
        """
        hover_text = []
        for _, row in df.iterrows():
            text = (
                f"Label: {row['label']}<br>"
                f"File: {row['pdf_name']}<br>"
                f"Page: {row['page']}<br>"
                f"Image: {row['image_path']}"
            )
            hover_text.append(text)
        return hover_text

    def create_scatter(
        self,
        df: pd.DataFrame,
        coords: np.ndarray,
        title: str,
        filename: str,
        color_column: str = "label",
    ) -> Path:
        """Generate and save an interactive scatter plot.

        Args:
            df: DataFrame containing labels and file metadata.
            coords: 2D projection coordinates.
            title: Title for the plot.
            filename: Output HTML filename.
            color_column: Column used to color points.

        Returns:
            Path to the saved HTML file.
        """
        plot_df = df.copy()
        plot_df["x"] = coords[:, 0]
        plot_df["y"] = coords[:, 1]
        plot_df["hover"] = self._build_hover_text(df)

        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color=color_column,
            hover_name="label",
            hover_data={"x": False, "y": False, color_column: False},
            custom_data=["hover"],
            text=None,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            title=title,
            opacity=0.8,
        )
        fig.update_traces(
            marker=dict(size=9, line=dict(width=0.6, color="white")),
            hovertemplate="%{customdata[0]}",
        )
        fig.update_layout(
            template="plotly_white",
            legend_title="Label",
            font=dict(family="Arial", size=12),
        )

        output_path = self.output_dir / filename
        fig.write_html(output_path, full_html=True, include_plotlyjs="cdn")
        return output_path


class CADAnalysisPipeline:
    """End-to-end pipeline for CAD PDF embedding and visualization."""

    def __init__(
        self,
        dataset_root: Path,
        image_cache_dir: Path,
        embedding_cache_dir: Path,
        output_dir: Path,
        model_name: str = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        device: str = "cpu",
        batch_size: int = 8,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        tsne_perplexity: float = 30.0,
        tsne_max_iter: int = 1000,
        random_state: int | None = 42,
        dpi: int = 300,
        allow_cpu_fallback: bool = True,
    ) -> None:
        """Set up the pipeline components.

        Args:
            dataset_root: Root directory that contains subfolders of PDFs per label.
            image_cache_dir: Directory used to store rendered PDF pages.
            embedding_cache_dir: Directory used to cache embedding `.npy` files.
            output_dir: Directory used to store visualization outputs.
            model_name: OpenCLIP model name for sentence-transformers.
            device: Torch device string ("cpu", "cuda", or "mps").
            batch_size: Batch size for embedding extraction.
            n_neighbors: Number of neighbors for UMAP.
            min_dist: Minimum distance for UMAP.
            tsne_perplexity: Perplexity for t-SNE.
            tsne_max_iter: Maximum iterations for t-SNE.
            random_state: Random seed for reproducibility.
            dpi: DPI used when rasterizing PDF pages.
            allow_cpu_fallback: Whether to fall back to CPU if GPU OOM occurs.
        """
        self.dataset_root = dataset_root
        self.image_cache_dir = image_cache_dir
        self.embedding_cache_dir = embedding_cache_dir
        self.output_dir = output_dir
        self.converter = PDFToImageConverter(image_cache_dir=image_cache_dir, dpi=dpi)
        self.extractor = OpenCLIPExtractor(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.reducer = ManifoldReducer(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=random_state,
            tsne_perplexity=tsne_perplexity,
            tsne_max_iter=tsne_max_iter,
        )
        self.visualizer = VisualizationBuilder(output_dir=output_dir)
        self.model_name = self.extractor.model_name_used
        self.random_state = random_state

        _ensure_directory(self.dataset_root)
        _ensure_directory(self.image_cache_dir)
        _ensure_directory(self.embedding_cache_dir)
        _ensure_directory(self.output_dir)

    def _scan_pdfs(self) -> list[Path]:
        """Find all PDF files under the dataset root.

        Returns:
            A sorted list of PDF file paths.
        """
        return sorted(self.dataset_root.rglob("*.pdf"))

    def _label_from_path(self, pdf_path: Path) -> str:
        """Derive label from the first directory component under the dataset root.

        Args:
            pdf_path: Path to a PDF file.

        Returns:
            Label name inferred from the directory structure.
        """
        relative = pdf_path.relative_to(self.dataset_root)
        return relative.parts[0] if relative.parts else "unknown"

    def _collect_images(self, pdf_paths: Iterable[Path]) -> list[dict[str, Path]]:
        """Convert PDFs to images and collect metadata.

        Args:
            pdf_paths: Iterable of PDF file paths.

        Returns:
            A list of metadata dictionaries for each rendered page.
        """
        records: list[dict[str, Path]] = []
        for pdf_path in pdf_paths:
            label = self._label_from_path(pdf_path)
            images = self.converter.convert_pdf(pdf_path, label)
            for page_index, image_path in enumerate(images, start=1):
                embed_path = self.embedding_cache_dir / image_path.relative_to(
                    self.image_cache_dir
                )
                embed_path = embed_path.with_suffix(".npy")
                records.append(
                    {
                        "label": label,
                        "pdf_path": pdf_path,
                        "image_path": image_path,
                        "embedding_path": embed_path,
                        "page": page_index,
                        "pdf_name": pdf_path.name,
                        "image_name": image_path.name,
                    }
                )
        return records

    def _load_cached_embedding(self, embedding_path: Path) -> np.ndarray | None:
        """Load cached embedding if available.

        Args:
            embedding_path: Path to the `.npy` embedding file.

        Returns:
            Loaded embedding array, or None if not present or invalid.
        """
        if not embedding_path.exists():
            return None
        try:
            return np.load(embedding_path)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to load cached embedding %s: %s", embedding_path, exc
            )
            return None

    def _save_embedding(self, embedding_path: Path, embedding: np.ndarray) -> None:
        """Persist embedding to disk.

        Args:
            embedding_path: Target `.npy` file path.
            embedding: Embedding vector to save.
        """
        _ensure_directory(embedding_path.parent)
        np.save(embedding_path, embedding)

    def _prepare_embeddings(
        self, records: list[dict[str, Path]]
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Load or compute embeddings with caching.

        Args:
            records: Metadata records containing image and embedding paths.

        Returns:
            A tuple of (DataFrame, stacked embeddings array).

        Raises:
            RuntimeError: If no embeddings can be produced.
        """
        embeddings: list[np.ndarray | None] = [None] * len(records)
        to_compute_indices: list[int] = []

        for idx, record in enumerate(records):
            cached = self._load_cached_embedding(record["embedding_path"])
            if cached is not None:
                embeddings[idx] = cached
            else:
                to_compute_indices.append(idx)

        if to_compute_indices:
            paths_to_compute = [
                records[idx]["image_path"] for idx in to_compute_indices
            ]
            new_embeddings = self.extractor.extract(paths_to_compute)
            for i, embedding in zip(to_compute_indices, new_embeddings):
                embeddings[i] = embedding
                self._save_embedding(records[i]["embedding_path"], embedding)

        final_records: list[dict[str, Path]] = []
        final_embeddings: list[np.ndarray] = []
        for record, embedding in zip(records, embeddings):
            if embedding is None:
                logger.warning(
                    "Skipping record without embedding: %s", record["image_path"]
                )
                continue
            final_records.append(record)
            final_embeddings.append(embedding)

        if not final_embeddings:
            raise RuntimeError(
                "No embeddings were generated. Check input data and cache paths."
            )

        df = pd.DataFrame(final_records)
        stacked_embeddings = np.stack(final_embeddings, axis=0)
        return df, stacked_embeddings

    def run(self) -> dict[str, object]:
        """Execute the full pipeline and return artifacts.

        Returns:
            Dictionary containing data frame, embeddings, projections, and plot paths.

        Raises:
            FileNotFoundError: If no PDFs are found in the dataset root.
            RuntimeError: If images or embeddings cannot be produced.
        """
        pdf_paths = self._scan_pdfs()
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found under {self.dataset_root}")

        image_records = self._collect_images(pdf_paths)
        if not image_records:
            raise RuntimeError("No images were generated from the PDFs.")

        df, embeddings = self._prepare_embeddings(image_records)

        umap_coords = self.reducer.run_umap(embeddings)
        tsne_coords = self.reducer.run_tsne(embeddings)

        umap_title = f"UMAP Projection - {self.model_name} | Samples: {len(df)}"
        tsne_title = f"t-SNE Projection - {self.model_name} | Samples: {len(df)}"

        umap_path = self.visualizer.create_scatter(
            df, umap_coords, umap_title, "umap_projection.html"
        )
        tsne_path = self.visualizer.create_scatter(
            df, tsne_coords, tsne_title, "tsne_projection.html"
        )

        artifacts: dict[str, object] = {
            "dataframe": df,
            "embeddings": embeddings,
            "umap_coords": umap_coords,
            "tsne_coords": tsne_coords,
            "umap_plot": umap_path,
            "tsne_plot": tsne_path,
        }
        return artifacts


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    pipeline = CADAnalysisPipeline(
        dataset_root=Path("./data/吉輔提供資料"),
        image_cache_dir=Path("./cache/images"),
        embedding_cache_dir=Path("./cache/embeddings"),
        output_dir=Path("./results/cad_analysis"),
        model_name="laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        device="cuda",
        batch_size=8,
        n_neighbors=20,
        min_dist=0.05,
        tsne_perplexity=35,
        random_state=42,
        dpi=300,
    )

    artifacts = pipeline.run()
    logger.info(
        "Artifacts generated: %s",
        {k: str(v) if isinstance(v, Path) else type(v) for k, v in artifacts.items()},
    )
