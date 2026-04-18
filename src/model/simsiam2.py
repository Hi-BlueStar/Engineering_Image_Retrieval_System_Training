"""
SimSiam 模型定義與核心功能庫
------------------------------------------------------------------------------
此模組實作了 SimSiam (Simple Siamese Representation Learning) 的核心組件，
並針對工程圖 (CAD/Line Art) 檢索進行了優化。

主要組件：
1. GeometryAwareTransform: 針對幾何線條特性的資料增強策略。
2. SimSiam: 模型架構 (Backbone + Projector + Predictor)。
3. Loss Function: 負餘弦相似度損失。
4. Utility Functions: 訓練與評估的原子操作。

References:
    Chen & He. "Exploring Simple Siamese Representation Learning". CVPR 2021.
"""

import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


# 防止讀取損壞圖片 (Truncated Images) 時報錯
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------------------------------------------------------
# 1. 資料增強與資料集 (Data Augmentation & Dataset)
# -----------------------------------------------------------------------------

# GeometryAwareTransform and make_inference_transform have been moved to augmentations.py


class UnlabeledImages(Dataset):
    """
    無標籤影像資料集，支援自動轉換為灰階。
    """

    def __init__(self, paths: list[Path], transform, grayscale: bool = True):
        """
        Args:
            paths (List[Path]): 圖片路徑列表。
            transform (Callable): 增強函數 (需產生兩視角)。
            grayscale (bool): 是否強制轉為單通道灰階圖。
        """
        self.paths = list(paths)
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i) -> tuple[torch.Tensor, torch.Tensor]:
        """讀取並回傳一張圖片的兩個增強版本。"""
        p = self.paths[i]
        try:
            img = Image.open(p)
            # 根據模式轉換圖片格式
            if self.grayscale:
                img = img.convert("L")  # L mode = 8-bit pixels, black and white
            else:
                img = img.convert("RGB")

            return self.transform(img)
        except Exception as e:
            # 遇到壞圖時的簡單處理：印出錯誤並拋出，建議在外部先做資料清洗
            print(f"Error loading {p}: {e}")
            # 簡單容錯：隨機生成一張雜訊圖避免崩潰 (實際專案建議先清洗資料)
            # 注意：這裡假設輸入大小為 512，若變更需同步
            dummy = Image.new("L" if self.grayscale else "RGB", (512, 512))
            return self.transform(dummy)


# -----------------------------------------------------------------------------
# 2. SimSiam 模型架構 (Model Architecture)
# -----------------------------------------------------------------------------

def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    bn_last: bool = True,
    dropout: float = 0.0,
) -> nn.Sequential:
    """輔助函數：建立多層感知機 (MLP) 區塊。

    通常用於 Projector 或 Predictor。結構為 Linear -> BN -> ReLU -> ...

    Args:
        in_dim: 輸入維度。
        hidden_dim: 隱藏層維度。
        out_dim: 輸出維度。
        bn_last: 最後一層是否加 BatchNorm。
        dropout: 第一層後的 Dropout 比率。

    Returns:
        nn.Sequential: 構建好的 MLP 模型。
    """
    layers = [
        nn.Linear(in_dim, hidden_dim, bias=False),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden_dim, out_dim, bias=False)]
    if bn_last:
        layers.append(nn.BatchNorm1d(out_dim, affine=True))
    return nn.Sequential(*layers)


class SimSiam(nn.Module):
    """SimSiam 模型主體，支援自定義輸入通道。

    結構：
        x -> Backbone -> f (features)
        f -> Projector -> z (embeddings)
        z -> Predictor -> p (predictions)

    機制：
        Loss = D(p1, stop_gradient(z2)) + D(p2, stop_gradient(z1))
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        proj_dim: int = 2048,
        pred_hidden: int = 512,
        dropout: float = 0.0,
        pretrained: bool = False,
        in_channels: int = 1,  # 支援灰階輸入
    ):
        """初始化模型。

        參數
            backbone (str): 'resnet18' 或 'resnet50'。
            proj_dim (int): Projector 輸出維度 (亦即 embedding 維度)。
            pred_hidden (int): Predictor 隱藏層維度 (瓶頸層)。
            dropout (float): Projector 中的 Dropout 機率。
            pretrained (bool): 是否載入 ImageNet 預訓練權重。
            in_channels (int): 輸入圖片的通道數 (預設 1 為灰階)。
        """
        super().__init__()

        # 1. 建立 Backbone
        if backbone == "resnet18":
            net = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        elif backbone == "resnet50":
            net = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            raise NotImplementedError(f"Unsupported backbone: {backbone}")

        # 修改第一層卷積以適應輸入通道 (如灰階圖)
        if in_channels != 3:
            # ResNet 的 conv1 原始結構是 (64, 3, 7, 7, stride=2, padding=3)，需改為 (64, in_channels, 7, 7)
            old_conv = net.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # 若載入預訓練權重，可將 RGB 權重平均化到單通道 (由使用者自行決定是否需要更複雜的遷移策略)
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    # Sum over channel dimension (dim 1) and divide by 3
                    # shape: [64, 3, 7, 7] -> [64, 1, 7, 7]
                    new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True) / 3.0

            net.conv1 = new_conv

        # 取得 Backbone 輸出特徵維度 (ResNet50 為 2048, ResNet18 為 512)
        feat_dim = net.fc.in_features
        # 移除原始分類用的全連接層 (fc)
        net.fc = nn.Identity()
        self.backbone = net

        # 2. 建立 Projector (投影頭)
        # SimSiam 論文建議 Projector 為 3 層 MLP，此處簡化為 2 層或依照 _mlp 實作
        # 輸入: feat_dim -> 隱藏: 2048 -> 輸出: proj_dim
        self.projector = _mlp(feat_dim, 2048, proj_dim, bn_last=True, dropout=dropout)

        # 3. 建立 Predictor (預測頭)
        # 這是 SimSiam 與其他 SSL 方法最大的不同點，用於匹配另一視角的輸出
        # 結構: proj_dim -> pred_hidden -> proj_dim
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_hidden, bias=False),
            nn.BatchNorm1d(pred_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(pred_hidden, proj_dim),
        )

        # 4. 權重初始化 (僅對 MLP 部分)（偏向小 std）
        # 固定 Backbone，對新增的 MLP 進行初始化 (Truncated Normal 效果較好)
        # Backbone 若是 pretrained 則不動，否則 PyTorch 已有預設初始化
        for m in list(self.projector.modules()) + list(self.predictor.modules()):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """前向傳播。

        參數:
            x1: 第一個視角的圖片 Batch [B, C, H, W]。
            x2: 第二個視角的圖片 Batch [B, C, H, W]。

        返回:
            p1, p2: Predictor 的預測向量。
            z1, z2: Projector 的目標投影向量 (已 detach，用於作為 Target)。
        """
        # 共享 Backbone 與 Projector
        f1 = self.backbone(x1)
        f2 = self.backbone(x2)

        z1 = self.projector(f1)
        z2 = self.projector(f2)

        # Predictor 轉換
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # 關鍵：回傳 detach 的 z，在 Loss 計算時作為常數目標 (Stop-Gradient)
        return p1, p2, z1.detach(), z2.detach()


def D(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """SimSiam 損失函數：負餘弦相似度 (Negative Cosine Similarity)。

    公式:
    $$ \\mathcal{L} = - \frac{p}{\\|p\\|_2} \\cdot \frac{z}{\\|z\\|_2} $$

    參數:
        p: Predictor 輸出的預測向量 [B, Dim]。
        z: Projector 輸出的目標投影向量，Target 投影向量 [B, Dim]。

    Returns:
        torch.Tensor: 純量 Loss (平均值)。
    """
    # 務必先做 L2 Normalize
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    # 點積後取平均並加負號
    return -(p * z).sum(dim=1).mean()


# -----------------------------------------------------------------------------
# 3. 訓練與評估工具 (Training Utilities)
# -----------------------------------------------------------------------------

def calculate_collapse_std(z: torch.Tensor) -> float:
    """計算 L2 正規化後特徵在各維度的標準差平均值，用於監控維度坍塌。
    
    理論上，一個完美均勻分佈的特徵空間，其維度標準差應趨近於 1/sqrt(d)。
    若此數值趨近於 0，表示模型僅在極少數維度上產生變化 (Dimensional Collapse)。
    """
    z_norm = F.normalize(z, dim=1)
    # 沿著 Batch 維度計算 std，然後對所有特徵維度取平均
    return z_norm.std(dim=0).mean().item()

def train_one_epoch(
    model: SimSiam,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str
) -> tuple[float, float]:
    """執行一個 Epoch 的訓練。

    參數:
        model: SimSiam 模型實例。
        loader: DataLoader，回傳 (v1, v2)。
        optimizer: 優化器 (AdamW 或 SGD)。
        scaler: GradScaler，用於混合精度訓練 (AMP)。
        device: 'cuda' 或 'cpu'。

    Returns:
        tuple[float, float]: 該 Epoch 的平均 Loss 與平均 Std。
    """
    model.train()
    total_loss = 0.0
    total_std = 0.0
    num_batches = 0

    # 判斷是否啟用 AMP
    use_amp = scaler is not None and str(device).startswith("cuda")
    # 兼容性處理：PyTorch 2.0+ 使用 torch.amp，舊版使用 torch.cuda.amp
    if hasattr(torch, "amp"):
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
    else:
        amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)

    for v1, v2 in loader:
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # set_to_none 稍微快一點

        with amp_ctx:
            # p1 預測 z2, p2 預測 z1
            p1, p2, z1, z2 = model(v1, v2)
            # 對稱 Loss：0.5 * (Loss(p1, z2) + Loss(p2, z1))
            loss = 0.5 * (D(p1, z2) + D(p2, z1))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # 計算特徵坍塌指標 (取 z1 與 z2 的平均標準差)
        with torch.no_grad():
            batch_std = (calculate_collapse_std(z1) + calculate_collapse_std(z2)) / 2.0
            total_std += batch_std

        num_batches += 1

    return total_loss / max(num_batches, 1), total_std / max(num_batches, 1)


@torch.no_grad()
def evaluate(model: SimSiam, loader: DataLoader, device: str) -> tuple[float, float]:
    """驗證模型損失與標準差 (注意：這不是下游分類準確率，僅是 SSL 收斂指標)。"""
    model.eval()
    total_loss = 0.0
    total_std = 0.0
    num_batches = 0

    for v1, v2 in loader:
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        p1, p2, z1, z2 = model(v1, v2)
        loss = 0.5 * (D(p1, z2) + D(p2, z1))
        total_loss += loss.item()
        
        batch_std = (calculate_collapse_std(z1) + calculate_collapse_std(z2)) / 2.0
        total_std += batch_std
        
        num_batches += 1

    return total_loss / max(num_batches, 1), total_std / max(num_batches, 1)
