"""
Debug Pipeline Script.
Visualizes intermediate steps of image processing: Binarization, Component Analysis, ROI Extraction.
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import sys

# Add src to path to allow imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

try:
    from src.image_preprocessing3 import auto_binarize, analyze_components, select_large_small
except ImportError:
    print("Error: Could not import src.image_preprocessing3")
    sys.exit(1)

def debug_visualize_processing(img_path: str, output_dir: str = "outputs/debug_vis"):
    """
    Run pipeline on an image and save visualization of each step.
    """
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(img_path)
    base_name = os.path.splitext(filename)[0]
    
    img = cv2.imread(img_path) # BGR
    if img is None:
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Setup plot: 1 Row, 4 Cols (Original, Binary, ROI Selection, Components)
    # Actually, let's do a 2x3 Grid to show more details
    # 1. Original
    # 2. Binary
    # 3. All Components BBox
    # 4. Top-N Selected BBox
    # 5. Cropped ROI 1
    # 6. Cropped ROI 2 ... (Might need flexible layout)
    
    top_n = 5
    
    # Step 1: Binarization
    bw01, bg_mode = auto_binarize(img)
    
    # Step 2: Analysis
    comps = analyze_components(bw01)
    
    # Step 3: Selection
    large_comps, _ = select_large_small(comps, top_n=top_n, remove_largest=False)
    
    # Create Figure
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)
    
    # Plot 1: Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title("1. Original Image")
    ax1.axis('off')
    
    # Plot 2: Binary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(bw01, cmap='gray')
    ax2.set_title(f"2. Binarized (Mode: {bg_mode})")
    ax2.axis('off')
    
    # Plot 3: All Components
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img_rgb)
    for comp in comps:
        x, y, w, h = comp.bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='blue', facecolor='none')
        ax3.add_patch(rect)
    ax3.set_title(f"3. All Components ({len(comps)})")
    ax3.axis('off')
    
    # Plot 4: Selected Top-5
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(img_rgb)
    for i, comp in enumerate(large_comps):
        x, y, w, h = comp.bbox
        # Red thick box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax4.add_patch(rect)
        ax4.text(x, y, str(i), color='red', fontsize=12, fontweight='bold')
    ax4.set_title(f"4. Top-{top_n} Selected")
    ax4.axis('off')
    
    # Plot 5-8: Cropped ROIs (First 4)
    for i, comp in enumerate(large_comps[:4]):
        x, y, w, h = comp.bbox
        roi = img_rgb[y:y+h, x:x+w]
        
        ax = fig.add_subplot(gs[1, i])
        if roi.size > 0:
            ax.imshow(roi)
        ax.set_title(f"ROI {i} (Area: {comp.area})")
        ax.axis('off')
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"debug_{base_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved debug visualization to {out_path}")

if __name__ == "__main__":
    test_image = "data/engineering_images_Clean_100dpi/換刀臂/175B30-DSV-L00-10130.png"
    debug_visualize_processing(test_image)
