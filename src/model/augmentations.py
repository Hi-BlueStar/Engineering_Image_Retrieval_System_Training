"""
SimSiam 工程圖資料增強模組 (Data Augmentation Module for SimSiam - Engineering Drawings)
------------------------------------------------------------------------------
本模組實作了專為工程圖 (Engineering Drawings, 例如 CAD, Line Art) 設計的進階資料增強策略，
重點在於：
1. 形態學擾動 (Morphological Perturbations)：針對線條粗細與拓撲結構的魯棒性。
2. 幾何與彈性變換 (Geometric & Elastic Transformations)：針對形狀與視角不變性。
3. 遮擋與結構雜訊 (Occlusion & Structural Noise)：增強對局部特徵缺失的魯棒性。
4. 測試時增強 (Test-Time Augmentation, TTA)：推論階段的固定或隨機變換。
"""

import random
import torch
import numpy as np
import torchvision.transforms as T
import PIL.Image as Image
import PIL.ImageFilter as ImageFilter
import cv2
import PIL.ImageOps as ImageOps
from typing import Tuple, List, Union

# -----------------------------------------------------------------------------
# 1. 形態學擾動 (Morphological Perturbations)
# -----------------------------------------------------------------------------

class RandomMorphology:
    """
    隨機應用膨脹 (Dilation) 或腐蝕 (Erosion) 以模擬線條粗細變化
    以及退化效果 (例如斷線或墨水暈開)。
    """
    def __init__(self, p: float = 0.5, kernel_range: Tuple[int, int] = (3, 5)):
        self.p = p
        self.kernel_range = kernel_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        # 決定操作：0 代表腐蝕 (Erosion, 變細)，1 代表膨脹 (Dilation, 變粗)
        # 注意：對於黑色背景上的白色線條：
        #   - MinFilter (Erosion) 會讓白色線條變細/消失
        #   - MaxFilter (Dilation) 會讓白色線條變粗
        # 對於白色背景上的黑色線條 (典型的 CAD)，則相反。
        # 假設這是典型的 PIL 用法，其中 0=黑，255=白。
        
        kernel_size = random.choice([k for k in range(self.kernel_range[0], self.kernel_range[1] + 1, 2)])
        op_type = random.choice(["dilation", "erosion"])

        # 檢查圖片模式 (L 代表 8-bit 像素，黑白)
        if img.mode != 'L':
            img = img.convert('L')

        if op_type == "dilation":
            # 擴張亮區 (若線條為白色) 或暗區 (若線條為黑色)？
            # 標準 ImageFilter 選項：
            # - MaxFilter: 選取最亮像素 -> 對於黑色背景上的白色物體為膨脹 (Dilation)
            # - MinFilter: 選取最暗像素 -> 對於黑色背景上的白色物體為腐蝕 (Erosion)
            return img.filter(ImageFilter.MaxFilter(kernel_size))
        else:
            return img.filter(ImageFilter.MinFilter(kernel_size))

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, kernel_range={self.kernel_range})"


# -----------------------------------------------------------------------------
# 2. 幾何與彈性變換 (Geometric & Elastic Transformations)
# -----------------------------------------------------------------------------

class RandomElasticTransform:
    """
    應用局部彈性變形 (Elastic Deformations)，類似水波紋效果。
    包裝 torchvision 的 ElasticTransform，但若有需要可處理 PIL/Tensor 轉換。
    """
    def __init__(self, alpha: float = 50.0, sigma: float = 5.0, p: float = 0.5):
        self.transform = T.RandomApply([
            T.ElasticTransform(alpha=alpha, sigma=sigma, interpolation=T.InterpolationMode.BILINEAR)
        ], p=p)

    def __call__(self, img):
        return self.transform(img)


# -----------------------------------------------------------------------------
# 3. 遮擋與結構雜訊 (Occlusion & Structural Noise)
# -----------------------------------------------------------------------------

class RandomCutout:
    """
    隨機移除矩形區域 (Cutout)。
    """
    def __init__(self, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0, p=0.5):
        # 注意：T.RandomErasing 需要 Tensor 輸入。
        # 若我們在 PIL 上使用，需要自訂實作。
        # 但我們會在 ToTensor 之後使用此類別。
        self.transform = T.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transform(tensor)


class SaltAndPepperNoise:
    """
    在 PIL 圖片中隨機將像素翻轉為黑色 (0) 或白色 (255)。
    """
    def __init__(self, p: float = 0.5, density: float = 0.05):
        self.p = p
        self.density = density  # 雜訊像素的百分比
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        if img.mode != 'L':
            img = img.convert('L')

        # 轉為 numpy 以加速操作
        data = np.array(img)
        h, w = data.shape
        
        # 雜訊像素總數
        n_noise = int(h * w * self.density)
        
        # 隨機索引
        inds = np.random.choice(h * w, n_noise, replace=False)
        rows, cols = np.unravel_index(inds, (h, w))
        
        # 撒鹽 (Salt, 255) 或撒胡椒 (Pepper, 0)？每個雜訊像素各 50% 機率
        noise_vals = np.random.choice([0, 255], size=n_noise)
        
        data[rows, cols] = noise_vals
        
        return Image.fromarray(data)


# -----------------------------------------------------------------------------
# Main Augmentation Pipeline (主要增強管線)
# -----------------------------------------------------------------------------

class EngineeringDrawingAugmentation:
    """
    針對工程圖 (Engineering Drawings) 的綜合資料增強管線。
    
    管線階段 (Pipeline Stages)：
    1. 基礎幾何 (Base Geometric)：隨機調整大小裁切 (RandomResizedCrop)
    2. 方向 (Orientation)：翻轉 (Flip H/V)
    3. 結構 (Structural)：仿射 (Affine - 旋轉, 剪切, 縮放) 與透視 (Perspective)
    4. 拓撲 (Topological)：彈性變換 (Elastic Transform - 水波紋)
    5. 形態學 (Morphological)：膨脹/腐蝕 (Dilation/Erosion - 線條粗細)
    6. 雜訊 (Noise)：椒鹽雜訊 (Salt & Pepper)
    7. 轉為張量 (Convert to Tensor)
    8. 遮擋 (Occlusion)：隨機移除 (Random Cutout / Erasing)
    9. 標準化 (Normalize)
    """
    def __init__(
        self,
        img_size: int = 512,
        mean: List[float] = [0.5],
        std: List[float] = [0.5]
    ):
        self.img_size = img_size
        
        # 1. & 2. & 3. 基礎幾何 (與原始 SimSiam 類似但經過調整)
        self.geometric_trans = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomApply([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5), # 工程圖通常具有對稱性
                T.RandomApply([
                    T.RandomAffine(
                        degrees=[-45, 45], # 引入更多的隨機性與拓撲改變，迫使建立全局不變性，避免網路依賴局部邊緣
                        interpolation=T.InterpolationMode.BILINEAR,
                    )
                ], p=0.3),
                T.RandomApply([
                    T.RandomAffine(
                        degrees=[-45, 45],
                        interpolation=T.InterpolationMode.BILINEAR,
                    )
                ], p=0.3),
            ],
            p=0.8)
        ])

        # 4. 彈性 (拓撲)
        # 注意：ElasticTransform 可作用於 Tensor 或 PIL (PyTorch 1.11+)。
        # 若是較新版的 torchvision，放在 Compose 中是安全的。
        self.elastic_trans = RandomElasticTransform(alpha=50.0, sigma=5.0, p=0.3)

        # 5. 形態學 (線條粗細) - 基於 PIL
        self.morph_trans = RandomMorphology(p=0.3, kernel_range=(3, 5))

        # 6. 雜訊 - 基於 PIL
        self.noise_trans = SaltAndPepperNoise(p=0.2, density=0.02)

        # 7. ToTensor & 9. Normalize
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        
        # 8. 遮擋 (Cutout) - 基於 Tensor
        self.cutout_trans = RandomCutout(scale=(0.02, 0.15), value=0, p=0.3)

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        產生同一張圖片的兩個增強視角。
        """
        def augment(img):
            # 1. 幾何與拓撲擾動 (PIL 階段)
            out = self.geometric_trans(img)
            
            # 重新啟用：形態學擾動 (模擬線條粗細變異與列印退化)
            out = self.morph_trans(out)

            out = self.elastic_trans(out)
            
            # 2. 張量轉換與標準化
            out = self.normalize(out)
            
            # 重新啟用：遮擋增強 (必須在 Tensor 階段執行)
            # 迫使模型在缺失局部特徵時，仍能推斷出全局特徵
            out = self.cutout_trans(out)
            return out

        v1 = augment(x)
        v2 = augment(x)
        return v1, v2

def make_inference_transform(
    img_size: int = 512,
    mean: List[float] = [0.5],
    std: List[float] = [0.5]
) -> T.Compose:
    """
    用於驗證/推論的確定性轉換 (無隨機增強)。
    """
    return T.Compose([
        T.Resize(int(img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

# -----------------------------------------------------------------------------
# Test-Time Augmentation (TTA) Strategies
# -----------------------------------------------------------------------------

class TTAHorizontalFlip:
    """
    水平翻轉
    TTA: Original + Horizontal Flip
    Returns List[Tensor] of size 2.
    """
    def __init__(self, base_transform: T.Compose):
        self.base_transform = base_transform

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        # View 1: Original (Processed)
        v1 = self.base_transform(img)
        
        # View 2: Horizontal Flip -> Processed
        # Note: We must flip BEFORE ToTensor/Normalize if base_transform includes them.
        # But base_transform usually does Resize->CenterCrop->ToTensor->Normalize.
        # So we should manually manipulate PIL image or inject Flip.
        # Strategy: Apply flip on PIL, then run base_transform.
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        v2 = self.base_transform(img_flipped)
        
        return [v1, v2]

class TTAVerticalFlip:
    """
    垂直翻轉
    TTA: Original + Vertical Flip
    Returns List[Tensor] of size 2.
    """
    def __init__(self, base_transform: T.Compose):
        self.base_transform = base_transform

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        # View 1: Original
        v1 = self.base_transform(img)
        
        # View 2: Vertical Flip
        img_flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        v2 = self.base_transform(img_flipped)
        
        return [v1, v2]

class TTAMultiScale:
    """
    多尺度縮放，此處「縮放」是指先調整影像大小，然後裁切至目標尺寸，相當於放大或縮小。
    TTA: Original (1.0x) + Scaled versions (e.g. 0.9x, 1.1x).
    Note: 'Scale' here means we resize the image then crop to target size,
    effectively zooming in or out.
    """
    def __init__(self, base_transform: T.Compose, scales: List[float] = [0.9, 1.0, 1.1], img_size: int = 224):
        self.scales = scales
        self.img_size = img_size
        # Extract normalization from base_transform if possible, or assume standard
        # For simplicity, we construct a fresh transform chain for each scale.
        # But we need to know mean/std.
        # We assume base_transform[-1] is Normalize.
        self.normalize = base_transform.transforms[-1]
        self.to_tensor = base_transform.transforms[-2] # Assuming ToTensor is 2nd to last

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        results = []
        for s in self.scales:
            # Calculate resize target
            # Standard inference resize is int(img_size * 1.14) ~ 256 for 224
            # If s=1.0, we want standard behavior.
            target_resize = int(self.img_size * 1.14 * s)
            
            t = T.Compose([
                T.Resize(target_resize, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.img_size),
                self.to_tensor,
                self.normalize
            ])
            results.append(t(img))
        return results

class TTAFiveCrop:
    """
    五塊裁切，
    TTA: 5-Crop (TopLeft, TopRight, BottomLeft, BottomRight, Center).
    """
    def __init__(self, img_size: int = 224, mean=[0.5], std=[0.5]):
        self.img_size = img_size
        self.five_crop = T.FiveCrop(img_size)
        self.norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        # We need a resize before crop usually
        self.preprocess = T.Resize(int(img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC)

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        # 1. Resize
        img = self.preprocess(img)
        # 2. 5-Crop (Returns tuple of PIL images)
        crops = self.five_crop(img) 
        # 3. Transform each
        return [self.norm(c) for c in crops]

# -----------------------------------------------------------------------------
# New TTA Strategies (Ablation Study Expansion)
# -----------------------------------------------------------------------------

class TTARotation:
    """
    旋轉
    TTA: Original + Rotations (e.g., -5, 5 degrees).
    Useful for scans that are slightly skewed.
    """
    def __init__(self, base_transform: T.Compose, degrees: List[float] = [-5, 5]):
        self.base_transform = base_transform
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        results = [self.base_transform(img)] # Original
        
        for deg in self.degrees:
            # Rotate PIL image
            # expand=False to keep original size (might crop corners)
            # fillcolor needs to match background, assume white (255) for now? 
            # Or use pad mechanism. Let's use simple rotate.
            img_rot = img.rotate(deg, resample=Image.BICUBIC, expand=False)
            results.append(self.base_transform(img_rot))
            
        return results

class TTARotation90:
    """
    旋轉90度
    TTA: Original + 90, 180, 270 Degree Rotations.
    Total 4 views.
    """
    def __init__(self, base_transform: T.Compose):
        self.base_transform = base_transform
        self.angles = [90, 180, 270]

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        results = [self.base_transform(img)] # 0 degrees
        
        for angle in self.angles:
            # Rotate PIL image (expand=True might change aspect ratio if non-square, 
            # but usually we want to keep content. For 90 deg steps, expand=True swaps W/H.
            # Since subsequent transform does Resize/Crop, this is fine).
            # Using expand=True to ensure no cropping of corners for 90 deg rotations.
            img_rot = img.rotate(angle, resample=Image.BICUBIC, expand=True)
            results.append(self.base_transform(img_rot))
            
        return results

class TTAColorJitter:
    """
    顏色抖動
    TTA: Original + Varied Brightness/Contrast.
    Useful for scans with different lighting or scanner settings.
    """
    def __init__(self, base_transform: T.Compose, brightness: float = 0.2, contrast: float = 0.2):
        self.base_transform = base_transform
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        results = [self.base_transform(img)] # Original
        
        # Augment 2 more versions? Or just one?
        # Let's generate 2 random jitters.
        for _ in range(2):
            img_jitter = self.jitter(img)
            results.append(self.base_transform(img_jitter))
            
        return results

class TTAGaussianBlur:
    """
    高斯模糊
    TTA: Original + Blurred versions.
    Simonulates out-of-focus scans or low resolution.
    """
    def __init__(self, base_transform: T.Compose, kernel_size: int = 5, sigma: float = 2.0):
        self.base_transform = base_transform
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        results = [self.base_transform(img)] # Original
        
        # Blurring happens on PIL or Tensor? 
        # T.GaussianBlur works on both.
        # But base_transform usually takes PIL.
        # So we blur PIL then transform.
        
        img_blur = self.blur(img)
        results.append(self.base_transform(img_blur))
        
        return results

class TTAMorphology:
    """
    形態學
    TTA: Original + Morphological variations (Erosion, Dilation).
    Useful for varying line thickness in engineering drawings.
    """
    def __init__(self, base_transform: T.Compose, kernel_size: int = 3):
        self.base_transform = base_transform
        self.kernel_size = kernel_size

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        results = [self.base_transform(img)] # Original

        # Ensure image is grayscale (L mode) for consistent morphology
        if img.mode != 'L':
             img_gray = img.convert('L')
        else:
             img_gray = img

        # Erosion (Thinner lines for white-on-black, Thicker for black-on-white)
        # Assuming black lines on white background (standard CAD):
        # MinFilter = Erosion (makes black lines thicker)
        # MaxFilter = Dilation (makes black lines thinner)
        
        # 1. Erode (Thicken lines)
        img_erode = img_gray.filter(ImageFilter.MinFilter(self.kernel_size))
        results.append(self.base_transform(img_erode))

        # 2. Dilate (Thin lines)
        img_dilate = img_gray.filter(ImageFilter.MaxFilter(self.kernel_size))
        results.append(self.base_transform(img_dilate))

        return results

class TTACLAHE:
    """
    對比度增強
    TTA: Original + CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast.
    """
    def __init__(self, base_transform: T.Compose, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.base_transform = base_transform
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        results = [self.base_transform(img)] # Original

        # Convert to numpy/OpenCV
        # PIL 'L' -> numpy array
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img
        
        img_np = np.array(img_gray)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_clahe_np = clahe.apply(img_np)
        
        # Convert back to PIL
        img_clahe = Image.fromarray(img_clahe_np)
        results.append(self.base_transform(img_clahe))

        return results

class TTAGaussianNoise:
    """
    高斯雜訊
    TTA: Original + Gaussian Noise injection.
    Tests robustness against sensor noise.
    """
    def __init__(self, base_transform: T.Compose, sigma: float = 0.05):
        self.base_transform = base_transform
        self.sigma = sigma # Standard deviation of noise (assuming normalized 0-1 or 0-255 scale)

    def __call__(self, img: Image.Image) -> List[torch.Tensor]:
        results = [self.base_transform(img)] # Original

        # We can add noise in PIL (numpy) domain
        img_np = np.array(img).astype(np.float32)
        
        # Generate noise
        noise = np.random.normal(0, self.sigma * 255, img_np.shape).astype(np.float32)
        
        # Add noise
        img_noisy_np = img_np + noise
        img_noisy_np = np.clip(img_noisy_np, 0, 255).astype(np.uint8)
        
        img_noisy = Image.fromarray(img_noisy_np, mode=img.mode)
        results.append(self.base_transform(img_noisy))

        return results
