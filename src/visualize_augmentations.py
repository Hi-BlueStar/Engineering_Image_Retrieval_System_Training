#!/usr/bin/env python3
import argparse
import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import PIL.Image as Image
import math

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir is the directory containing this file
src_dir = current_dir
# project_root is the parent of src
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from model.augmentations import (
    EngineeringDrawingAugmentation,
    make_inference_transform,
    TTAHorizontalFlip,
    TTAVerticalFlip,
    TTAMultiScale,
    TTAFiveCrop,
    TTARotation,
    TTARotation90,
    TTAColorJitter,
    TTAGaussianBlur,
    TTAMorphology,
    TTACLAHE,
    TTAGaussianNoise
)

def denormalize(tensor, mean=[0.5], std=[0.5]):
    """
    Denormalize a tensor image
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

def tensor_to_numpy(tensor):
    img = denormalize(tensor).squeeze().numpy()
    img = np.clip(img, 0, 1)
    return img

def visualize_strategy(name, augmented_views, output_dir, original_img=None):
    """
    Visualizes the results of a TTA strategy.
    
    Args:
        name (str): Name of the strategy.
        augmented_views (List[torch.Tensor]): List of augmented view tensors.
        output_dir (str): Directory to save the plot.
        original_img (PIL.Image): Original image for reference (optional).
    """
    n_views = len(augmented_views)
    cols = n_views
    if original_img:
        cols += 1
        
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        axes = [axes]
    
    idx = 0
    if original_img:
        axes[idx].imshow(original_img, cmap='gray')
        axes[idx].set_title("Original Input")
        axes[idx].axis('off')
        idx += 1
        
    for i, view_tensor in enumerate(augmented_views):
        img_np = tensor_to_numpy(view_tensor)
        axes[idx].imshow(img_np, cmap='gray')
        axes[idx].set_title(f"{name}\nView {i+1}")
        axes[idx].axis('off')
        idx += 1
        
    plt.tight_layout()
    # Sanitize name for filename
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    save_path = os.path.join(output_dir, f"viz_{safe_name}.png")
    plt.savefig(save_path)
    print(f"Saved visualization for {name} to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Engineering Architecture Augmentations")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="results/visualize_augmentations", help="Directory to save visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Configuration
    mean = [0.5]
    std = [0.5]
    img_size = 512

    print(f"Loading image from: {args.image_path}")
    # Load image
    try:
        img = Image.open(args.image_path).convert('L')
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Base transform for TTA uses
    # Note: make_inference_transform includes Resize, CenterCrop, ToTensor, Normalize
    anchor_transform = make_inference_transform(img_size=img_size, mean=mean, std=std)

    # =========================================================================
    # Define all strategies with adjustable parameters here
    # =========================================================================
    strategies = {
        # 1. Main Training Augmentation Pipeline
        "Training_Pipeline": EngineeringDrawingAugmentation(img_size=img_size, mean=mean, std=std),

        # 2. TTA Strategies
        "TTA_HorizontalFlip": TTAHorizontalFlip(anchor_transform),
        "TTA_VerticalFlip": TTAVerticalFlip(anchor_transform),
        
        # Adjustable scales
        "TTA_MultiScale_Standard": TTAMultiScale(anchor_transform, scales=[0.9, 1.0, 1.1], img_size=img_size),
        "TTA_MultiScale_Wide": TTAMultiScale(anchor_transform, scales=[0.5, 1.0, 1.5], img_size=img_size),
        
        "TTA_FiveCrop": TTAFiveCrop(img_size=img_size, mean=mean, std=std),
        
        # Adjustable rotations
        "TTA_Rotation_Small": TTARotation(anchor_transform, degrees=[-5, 5]),
        "TTA_Rotation_Large": TTARotation(anchor_transform, degrees=[-15, 15]),
        "TTA_Rotation_Specific": TTARotation(anchor_transform, degrees=[30, 60]),
        
        "TTA_Rotation90": TTARotation90(anchor_transform),
        
        # Adjustable ColorJitter (requires converting to RGB inside if necessary, but TTAColorJitter handles L mode? 
        # torchvision ColorJitter supports L mode since relatively recent versions)
        "TTA_ColorJitter_Subtle": TTAColorJitter(anchor_transform, brightness=0.1, contrast=0.1),
        "TTA_ColorJitter_Strong": TTAColorJitter(anchor_transform, brightness=0.4, contrast=0.4),
        
        # Adjustable Blur
        "TTA_GaussianBlur_Mild": TTAGaussianBlur(anchor_transform, kernel_size=3, sigma=1.0),
        "TTA_GaussianBlur_Strong": TTAGaussianBlur(anchor_transform, kernel_size=9, sigma=3.0),
        
        # Adjustable Morphology
        "TTA_Morphology_k3": TTAMorphology(anchor_transform, kernel_size=3),
        "TTA_Morphology_k5": TTAMorphology(anchor_transform, kernel_size=5),
        
        # Adjustable CLAHE
        "TTA_CLAHE_Default": TTACLAHE(anchor_transform, clip_limit=2.0, tile_grid_size=(8, 8)),
        "TTA_CLAHE_HighContrast": TTACLAHE(anchor_transform, clip_limit=4.0, tile_grid_size=(4, 4)),
        
        # Adjustable Noise
        "TTA_GaussianNoise_Low": TTAGaussianNoise(anchor_transform, sigma=0.02),
        "TTA_GaussianNoise_High": TTAGaussianNoise(anchor_transform, sigma=0.10),
    }

    print(f"Visualizing {len(strategies)} strategies...")

    for name, strategy in strategies.items():
        print(f"Processing {name}...")
        try:
            # Special handling for Training Pipeline which returns a tuple of 2 views
            augmented_views = []
            if name == "Training_Pipeline":
                # It returns (v1, v2)
                views = strategy(img)
                augmented_views = list(views)
            else:
                # TTA strategies return List[Tensor]
                augmented_views = strategy(img)

            visualize_strategy(name, augmented_views, args.output_dir, img)
                
        except Exception as e:
            print(f"Failed to visualize {name}: {e}")
            import traceback
            traceback.print_exc()

    print("Done!")

if __name__ == "__main__":
    main()


# uv run python src/visualize_augmentations.py --image_path dataset/Run_01_Seed_42/Component_Dataset/train/2L0T-LB50012-1070000_large_L32_area166050_pad2.png --output_dir results/visualize_augmentations