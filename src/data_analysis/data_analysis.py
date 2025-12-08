"""
CAD Analysis Module
===================

This module provides an end-to-end pipeline for analyzing engineering CAD PDF drawings.
It includes functionality for:
1. Converting PDFs to high-resolution images.
2. Extracting semantic features using a SOTA OpenCLIP model.
3. Reducing dimensions using UMAP and t-SNE.
4. Generating interactive visualizations.

Author: Antigravity
Date: 2025-12-04
"""

import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm


# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CADPreprocessor:
    """
    Handles loading of CAD files (PDFs) and converting them to images.
    Implements caching to avoid redundant processing.
    """

    def __init__(self, cache_dir: str = ".cache_cad_images", dpi: int = 300):
        """
        Args:
            cache_dir: Directory to store converted images.
            dpi: DPI for PDF to Image conversion.
        """
        self.cache_dir = Path(cache_dir)
        self.dpi = dpi
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_hash(self, file_path: Path) -> str:
        """Calculates MD5 hash of a file for caching purposes."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def convert_pdf_to_image(self, pdf_path: Path) -> Path | None:
        """
        Converts the first page of a PDF to a PNG image.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Path to the converted image, or None if conversion failed.
        """
        try:
            file_hash = self._get_file_hash(pdf_path)
            output_filename = f"{pdf_path.stem}_{file_hash}.png"
            output_path = self.cache_dir / output_filename

            if output_path.exists():
                logger.debug(f"Cache hit for {pdf_path.name}")
                return output_path

            doc = fitz.open(pdf_path)
            if doc.page_count < 1:
                logger.warning(f"PDF {pdf_path.name} has no pages.")
                return None

            page = doc.load_page(0)  # Load first page
            pix = page.get_pixmap(dpi=self.dpi)
            pix.save(str(output_path))
            logger.info(f"Converted {pdf_path.name} to image.")
            return output_path

        except Exception as e:
            logger.error(f"Failed to convert {pdf_path}: {e}")
            return None

    def scan_directory(self, root_dir: str) -> pd.DataFrame:
        """
        Scans a directory for PDFs, converts them, and returns a DataFrame.

        Args:
            root_dir: Root directory containing subfolders as labels.

        Returns:
            DataFrame with columns ['filename', 'label', 'image_path'].
        """
        data = []
        root_path = Path(root_dir)

        if not root_path.exists():
            raise FileNotFoundError(f"Directory {root_dir} not found.")

        categories = [d for d in root_path.iterdir() if d.is_dir()]

        logger.info(
            f"Found {len(categories)} categories: {[c.name for c in categories]}"
        )

        for category_dir in categories:
            label = category_dir.name
            pdf_files = list(category_dir.glob("*.pdf"))

            logger.info(f"Processing {len(pdf_files)} files in category '{label}'...")

            for pdf_file in tqdm(pdf_files, desc=f"Converting {label}"):
                image_path = self.convert_pdf_to_image(pdf_file)
                if image_path:
                    data.append(
                        {
                            "filename": pdf_file.name,
                            "label": label,
                            "image_path": str(image_path),
                            "original_path": str(pdf_file),
                        }
                    )

        df = pd.DataFrame(data)
        logger.info(f"Total valid samples processed: {len(df)}")
        return df


class OpenCLIPExtractor:
    """
    Extracts semantic features from images using OpenCLIP models via SentenceTransformers.
    """

    def __init__(self, model_name: str = "clip-ViT-L-14"):
        """
        Args:
            model_name: Name of the SentenceTransformer model.
                        Default is a strong generic CLIP model.
                        For specific LAION-2B model, use 'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
                        if supported by sentence-transformers directly, or use the mapping.
        """
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info(f"Loading OpenCLIP model '{model_name}' on {self.device}...")

        # Note: sentence-transformers handles the specific CLIP model loading details
        self.model = SentenceTransformer(model_name, device=self.device)
        logger.info("Model loaded successfully.")

    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of image paths into embeddings.
        Processes images in batches to avoid OOM errors.

        Args:
            image_paths: List of file paths to images.
            batch_size: Batch size for inference and loading.

        Returns:
            Numpy array of shape (N_samples, Embedding_Dim).
        """
        all_embeddings = []

        logger.info(f"Encoding {len(image_paths)} images in batches of {batch_size}...")

        # Process in chunks
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch Processing"):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []
            valid_indices = []

            # Load batch
            for idx, path in enumerate(batch_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                    valid_indices.append(idx)
                except Exception as e:
                    logger.error(f"Error loading image {path}: {e}")

            if not batch_images:
                continue

            # Encode batch
            try:
                batch_embeddings = self.model.encode(
                    batch_images,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error encoding batch starting at index {i}: {e}")

            # Explicitly clear batch images to free memory
            del batch_images

        if not all_embeddings:
            raise ValueError("No valid images to encode.")

        return np.vstack(all_embeddings)


class ManifoldReducer:
    """
    Performs dimensionality reduction using UMAP and t-SNE.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit_transform_umap(
        self, embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1
    ) -> np.ndarray:
        """
        Applies UMAP reduction.

        Args:
            embeddings: Input feature vectors.
            n_neighbors: UMAP parameter for local/global structure balance.
            min_dist: UMAP parameter for point clustering tightness.

        Returns:
            2D array of reduced coordinates.
        """
        logger.info(
            f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, metric='cosine')..."
        )
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",  # Critical for high-dim embeddings like CLIP
            random_state=self.random_state,
            n_jobs=-1,
        )
        return reducer.fit_transform(embeddings)

    def fit_transform_tsne(
        self, embeddings: np.ndarray, perplexity: int = 30
    ) -> np.ndarray:
        """
        Applies t-SNE reduction.

        Args:
            embeddings: Input feature vectors.
            perplexity: t-SNE perplexity parameter.

        Returns:
            2D array of reduced coordinates.
        """
        # Adjust perplexity if N < perplexity
        n_samples = embeddings.shape[0]
        if n_samples < perplexity:
            perplexity = max(1, n_samples // 2)
            logger.warning(
                f"Sample size ({n_samples}) small. Adjusting perplexity to {perplexity}."
            )

        logger.info(f"Running t-SNE (perplexity={perplexity})...")
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric="cosine",
            init="pca",
            learning_rate="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        return tsne.fit_transform(embeddings)


class InteractiveVisualizer:
    """
    Generates interactive scatter plots using Plotly.
    """

    @staticmethod
    def plot_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        hover_data: list[str],
        title: str,
        output_path: str,
    ):
        """
        Creates and saves an interactive scatter plot.

        Args:
            df: DataFrame containing coordinates and metadata.
            x_col: Column name for X axis.
            y_col: Column name for Y axis.
            color_col: Column name for color coding (usually label).
            hover_data: List of columns to show on hover.
            title: Chart title.
            output_path: Path to save the HTML file.
        """
        logger.info(f"Generating plot: {title}")
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            hover_data=hover_data,
            title=title,
            template="plotly_dark",
            width=1200,
            height=800,
        )

        # Improve aesthetics
        fig.update_traces(
            marker=dict(size=8, opacity=0.8, line=dict(width=1, color="DarkSlateGrey"))
        )
        fig.update_layout(legend_title_text="Category")

        fig.write_html(output_path)
        logger.info(f"Saved plot to {output_path}")


class CADAnalysisPipeline:
    """
    Orchestrates the entire CAD analysis workflow.
    """

    def __init__(self, root_dir: str, output_dir: str = "output"):
        self.root_dir = root_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.preprocessor = CADPreprocessor()
        # Using a high-quality CLIP model compatible with sentence-transformers
        self.extractor = OpenCLIPExtractor(model_name="clip-ViT-L-14")
        self.reducer = ManifoldReducer()
        self.visualizer = InteractiveVisualizer()

    def run(self):
        logger.info("Starting CAD Analysis Pipeline...")

        # 1. Load and Preprocess
        df = self.preprocessor.scan_directory(self.root_dir)
        if df.empty:
            logger.error("No data found. Exiting.")
            return

        # 2. Extract Features
        # Check if we have cached embeddings
        embeddings_path = self.output_dir / "embeddings.npy"
        if embeddings_path.exists():
            logger.info("Loading cached embeddings...")
            embeddings = np.load(embeddings_path)
            if len(embeddings) != len(df):
                logger.warning("Cached embeddings count mismatch. Re-computing...")
                embeddings = self.extractor.encode_images(df["image_path"].tolist())
                np.save(embeddings_path, embeddings)
        else:
            embeddings = self.extractor.encode_images(df["image_path"].tolist())
            np.save(embeddings_path, embeddings)

        logger.info(f"Embeddings shape: {embeddings.shape}")

        # 3. Dimensionality Reduction
        # UMAP
        umap_coords = self.reducer.fit_transform_umap(embeddings)
        df["umap_x"] = umap_coords[:, 0]
        df["umap_y"] = umap_coords[:, 1]

        # t-SNE
        tsne_coords = self.reducer.fit_transform_tsne(embeddings)
        df["tsne_x"] = tsne_coords[:, 0]
        df["tsne_y"] = tsne_coords[:, 1]

        # 4. Visualization
        self.visualizer.plot_scatter(
            df,
            "umap_x",
            "umap_y",
            "label",
            ["filename"],
            f"CAD Drawings UMAP Projection (Model: CLIP-ViT-L-14, N={len(df)})",
            str(self.output_dir / "umap_projection.html"),
        )

        self.visualizer.plot_scatter(
            df,
            "tsne_x",
            "tsne_y",
            "label",
            ["filename"],
            f"CAD Drawings t-SNE Projection (Model: CLIP-ViT-L-14, N={len(df)})",
            str(self.output_dir / "tsne_projection.html"),
        )

        logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    # Example Usage
    # Ensure you have a directory structure like:
    # data/
    #   gearbox/
    #     drawing1.pdf
    #   piston/
    #     drawing2.pdf

    # Create dummy data for demonstration if 'data' folder doesn't exist
    dummy_data_path = Path("./data/吉輔提供資料")

    # Run Pipeline
    # Replace 'data_samples' with your actual data directory
    pipeline = CADAnalysisPipeline(
        root_dir="./data/吉輔提供資料", output_dir="analysis_results"
    )

    # Only run if there is data
    if any(dummy_data_path.iterdir()):
        try:
            pipeline.run()
        except Exception as e:
            logger.error(
                f"Pipeline run failed (likely due to empty dummy folders): {e}"
            )
            logger.info(
                "Please populate 'data_samples' with actual PDF files to see results."
            )
