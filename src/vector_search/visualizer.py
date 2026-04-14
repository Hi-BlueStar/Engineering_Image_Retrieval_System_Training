"""
Visualization Module for Retrieval Results.
Generating academic-style figures for retrieval evaluation.
"""
import os
import cv2
import logging
import numpy as np
import matplotlib
# Use Agg backend for non-interactive saving
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class RetrievalVisualizer:
    """
    Visualizer for Engineering Image Retrieval System.
    Generates comparison plots between Query and Top-K results.
    """

    def __init__(self, output_dir: str = "outputs/retrieval_vis"):
        """
        Initialize the visualizer.

        Args:
            output_dir (str): Directory to save visualization results.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize(self, query_img_path: str, results: List[Dict[str, Any]], query_image: Optional[np.ndarray] = None, top_k: int = 20, filename: str = "result.png"):
        """
        Visualize query image and its top-k retrieved results.

        Args:
            query_img_path (str): Path to the query image (used for title/label).
            results (List[Dict]): List of retrieval results from RetrievalEngine.
            query_image (np.ndarray, optional): The actual query image array. If None, reads from path.
            top_k (int): Number of top results to show.
            filename (str): Output filename.
        """
        if query_image is None:
            if os.path.exists(query_img_path):
                # Use cv2 to read (handling paths) but convert to RGB for plotting
                query_image = cv2.imread(query_img_path)
                if query_image is not None:
                    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
            else:
                logger.error(f"Query image not found: {query_img_path}")
                return

        # Prepare retrieval images
        retrieved_images = []
        scores = []
        ids = []

        for res in results[:top_k]:
            scores.append(res['score'])
            ids.append(res['parent_pdf_id'])
            
            # Retrieve path from metadata of the first matched component
            # Note: The result might aggregate multiple components, but usually they belong to the same file.
            # We assume 'details' or similar metadata contains the path.
            # However, 'details' in engine.py result is a List of metadatas.
            # We pick the first one to find the original image path.
            
            # Since our engine.py 'collect group by parent_pdf_id', we store 'details' key with list of metadatas.
            details = res.get('details', [])
            img_path = None
            if details:
                # Try to get 'path' from metadata.
                # In indexer.py we saved 'path'.
                # But 'details' might only contain list of metadatas from upserted vectors.
                # Let's check the first item.
                img_path = details[0].get('path')
            
            img = None
            if img_path and os.path.exists(img_path):
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            if img is None:
                # Placeholder for missing image
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                img.fill(200) # Gray
            
            retrieved_images.append(img)

        # Plotting
        # Layout: Query Image on the left (or top), Retrieved images in a grid.
        # For Top-20, a 4x5 or 5x4 grid for results is good.
        # Let's do:
        # Row 1: Query Image (Centered or Left)
        # Row 2-5: 5 Results per row.
        
        cols = 5
        rows = (top_k + cols - 1) // cols + 1 # +1 for Query row
        
        plt.figure(figsize=(15, 4 * rows))
        
        # Plot Query
        # Span the query across the first row center
        # Or just put it in the first subplot
        ax_q = plt.subplot2grid((rows, cols), (0, 0), colspan=2)
        if query_image is not None:
            ax_q.imshow(query_image)
        ax_q.set_title(f"Query: {os.path.basename(query_img_path)}", fontsize=12, fontweight='bold')
        ax_q.axis('off')
        
        # Text Info
        ax_text = plt.subplot2grid((rows, cols), (0, 2), colspan=3)
        ax_text.text(0.1, 0.5, f"Retrieval Results (Top-{top_k})\nStrategy: Multi-Component Aggregation", fontsize=14, va='center')
        ax_text.axis('off')

        # Plot Results
        for i, (img, score, pdf_id) in enumerate(zip(retrieved_images, scores, ids)):
            row_idx = (i // cols) + 1
            col_idx = i % cols
            
            ax = plt.subplot2grid((rows, cols), (row_idx, col_idx))
            ax.imshow(img)
            ax.set_title(f"Rank {i+1}\nID: {pdf_id}\nScore: {score:.4f}", fontsize=9)
            ax.axis('off')
            
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=100)
        plt.close()
        logger.info(f"Visualization saved to {out_path}")
