"""
SimSiam (PyTorch 版)
- 提供最小可用的模型/擴增/訓練迴圈與推論工具
- 可選 backbone：resnet18/resnet50（from torchvision）
- 主要介面：
  - SimSiam: 模型（backbone + projector + predictor）
  - TransformTwice: 資料擴增（回傳兩個視角）
  - UnlabeledImages: 無標籤影像資料集（配合 TransformTwice）
  - D: SimSiam 負餘弦相似度損失函數
  - train_one_epoch / evaluate: 訓練/驗證
  - embed_images / similarity_between_two_images: 取向量/兩張圖相似度

針對工程圖檢索優化的主要改動：
1. 資料增強：移除對線條有害的高斯模糊與顏色抖動，引入仿射變換與透視變換以增強幾何不變性。
2. 模型結構：支援單通道 (Grayscale) 輸入，適配工程圖格式。
3. 記憶體優化：重構推論函數，避免大量圖片載入導致的 OOM。

References:
    Chen & He. "Exploring Simple Siamese Representation Learning". CVPR 2021.

使用範例（以資料夾內所有圖片自監督訓練）：

    from pathlib import Path
    from PIL import Image
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    import torch

    from src.image_analysis.simsiam import (
        SimSiam, TransformTwice, UnlabeledImages, make_norm,
        train_one_epoch, evaluate
    )

    img_size = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    paths = [p for p in Path('data/images').rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}]

    # 可用 ImageNet 統計
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    norm = make_norm(mean, std)
    ds = UnlabeledImages(paths, transform=TransformTwice(img_size), norm=norm)
    dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

    model = SimSiam(backbone='resnet18', pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=(device.startswith('cuda')))
    for epoch in range(1, 11):
        loss = train_one_epoch(model, dl, opt, scaler, device)
        print(f"epoch {epoch} loss={loss:.4f}")
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


# 防止讀取損壞圖片時報錯
ImageFile.LOAD_TRUNCATED_IMAGES = True


# -----------------------------------------------------------------------------
# Geometry-Aware Augmentation / Dataset
# -----------------------------------------------------------------------------


class GeometryAwareTransform:
    """針對工程圖 (CAD/Line Art) 設計的幾何敏感資料增強類別。

    不同於自然影像，工程圖依賴線條與拓撲結構。
    此類別移除了模糊與色彩抖動，並增強了對旋轉、透視與變形的魯棒性。

    增強策略包含：
    1. RandomResizedCrop: 學習局部與全域的對應關係。
    2. RandomAffine: 模擬圖紙的旋轉、平移與剪切。
    3. RandomPerspective: 模擬掃描或拍攝時的視角偏差。
    4. RandomFlip: 學習鏡像不變性。
    """

    def __init__(
        self, img_size: int = 224, mean: list[float] = [0.5], std: list[float] = [0.5]
    ):
        """初始化幾何增強策略。

        參數:
            img_size (int): 輸出的圖片大小。
            mean (List[float]): 標準化均值 (建議單通道使用 [0.5])。
            std (List[float]): 標準化標準差 (建議單通道使用 [0.5])。
        """
        self.img_size = img_size
        self._build()

    def _build(self) -> None:
        """構建 Transform 管道。"""
        # 定義幾何變換序列
        self.transform = T.Compose(
            [
                # 隨機裁切：CAD 圖可能在不同縮放比例下具有相似特徵
                # scale 下限設為 0.4 避免切得太碎導致丟失整體幾何結構
                T.RandomResizedCrop(
                    self.img_size,
                    scale=(0.4, 1.0),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                # 幾何變換：隨機水平與垂直翻轉 (零件常具對稱性)
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # 關鍵：隨機仿射變換 (旋轉 +/- 90度, 平移, 縮放, 剪切)
                # 模擬圖紙不正、偏移或比例不一的情況
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=90,
                            translate=(0.1, 0.1),
                            scale=(0.8, 1.2),
                            shear=10,
                            interpolation=T.InterpolationMode.BILINEAR,
                        )
                    ],
                    p=0.5,
                ),
                # 關鍵：透視變換 (模擬圖紙未放平的情況)
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                # 轉為 Tensor 並標準化
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, x: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """對輸入圖片應用兩次獨立的幾何增強。

        參數:
        x (Image.Image): PIL Image 物件 (建議預先轉為 Grayscale)。

        返回:
        Tuple[torch.Tensor, torch.Tensor]: 兩個增強後的視角張量。
        """
        # 注意：若輸入為 PIL Image，transform 會自動處理
        v1 = self.transform(x)
        v2 = self.transform(x)
        return v1, v2


def make_inference_transform(
    img_size: int = 224, mean: list[float] = [0.5], std: list[float] = [0.5]
) -> T.Compose:
    """建立推論用的確定性 Transform (無隨機增強)。

    參數:
        img_size (int): 圖片大小。
        mean (List[float]): 標準化均值。
        std (List[float]): 標準化標準差。

    返回:
        T.Compose: 轉換組合 (Resize -> CenterCrop -> ToTensor -> Normalize)。
    """
    return T.Compose(
        [
            T.Resize(int(img_size * 1.14), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class UnlabeledImages(Dataset):
    """無標籤影像資料集，支援自動轉換為灰階。

    屬性:
        paths (List[Path]): 圖片路徑列表。
        transform (Callable): 增強函數 (產生兩視角)。
        grayscale (bool): 是否強制轉為單通道灰階圖。
    """

    def __init__(self, paths: list[Path], transform, grayscale: bool = True):
        self.paths = list(paths)
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
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
            raise e


# -----------------------------------------------------------------------------
# SimSiam Model
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
                bias=old_conv.bias,
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
        for m in [self.projector, self.predictor]:
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
# Train / Eval Utilities
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: SimSiam, loader: DataLoader, optimizer, scaler, device: str
) -> float:
    """執行一個 Epoch 的訓練。

    Args:
        model: SimSiam 模型實例。
        loader: DataLoader，回傳 (v1, v2)。
        optimizer: 優化器 (AdamW 或 SGD)。
        scaler: GradScaler，用於混合精度訓練 (AMP)。
        device: 'cuda' 或 'cpu'。

    Returns:
        float: 該 Epoch 的平均 Loss。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 判斷是否啟用 AMP
    use_amp = scaler is not None and str(device).startswith("cuda")
    # 兼容舊版與新版 PyTorch 的 autocast
    if hasattr(torch, "amp"):
        amp_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else torch.no_grad()
        )  # no_grad is dummy here
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
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model: SimSiam, loader: DataLoader, device: str) -> float:
    """驗證模型損失 (注意：這不是下游分類準確率，僅是 SSL 收斂指標)。"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for v1, v2 in loader:
        v1 = v1.to(device, non_blocking=True)
        v2 = v2.to(device, non_blocking=True)
        p1, p2, z1, z2 = model(v1, v2)
        loss = 0.5 * (D(p1, z2) + D(p2, z1))
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# -----------------------------------------------------------------------------
# Efficient Inference Utilities
# -----------------------------------------------------------------------------


class _InferenceDataset(Dataset):
    """內部類別：用於將路徑列表封裝為 Dataset，供推論 DataLoader 使用。"""

    def __init__(self, paths: list[Path], transform, grayscale: bool = True):
        self.paths = paths
        self.transform = transform
        self.grayscale = grayscale

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            img = Image.open(path)
            if self.grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            return self.transform(img)
        except Exception:
            # 回傳 None 標記失敗，由 collate_fn 或外部迴圈處理
            # 此處簡單處理：回傳一個全零的 Dummy Tensor (須確保維度與正常圖片一致)
            # 這裡我們假設 transform 最終會輸出 [C, H, W]
            # 若發生錯誤，後續處理需具備過濾機制
            return torch.zeros(1, 1, 1)  # 維度將在 DataLoader 中被檢測


# -----------------------------------------------------------------------------
# Embed / Similarity
# -----------------------------------------------------------------------------


@torch.no_grad()
def embed_images(
    model: SimSiam,
    paths: list[Path],
    transform: T.Compose,
    device: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> torch.Tensor:
    """提取圖片特徵向量 (記憶體優化版)。

    使用 DataLoader 進行批次讀取與推論，避免一次性將所有圖片載入記憶體。

    使用 Backbone + Projector 輸出特徵 (z)。
    注意：回傳前會進行 L2 Normalize，方便後續直接用 Dot Product 計算 Cosine Similarity。


    參數:
        model (SimSiam): 已載入權重的 SimSiam 模型。
        paths (List[Path]): 圖片路徑列表。
        transform (T.Compose): 推論用的前處理 (Resize -> CenterCrop -> Norm)。
        device (str): 運算裝置。
        batch_size (int): 批次大小。
        num_workers (int): DataLoader 的 worker 數量。

    返回:
        torch.Tensor: 所有圖片的特徵矩陣，形狀為 [N, dim] 的特徵矩陣。
    """
    model.eval()
    # 判斷輸入是否為灰階 (根據 transform 的 Normalize 參數或模型設定推斷較難，
    # 這裡假設使用者在呼叫端已配置好 transform，Dataset 僅負責讀圖)
    # 為保險起見，我們檢查 model.backbone.conv1.in_channels
    grayscale = model.backbone.conv1.in_channels == 1

    dataset = _InferenceDataset(paths, transform, grayscale=grayscale)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    embs_list = []

    for imgs in loader:
        # 過濾讀取錯誤的圖片 (簡單檢查維度)
        if imgs.dim() < 4:
            # 若發生錯誤，填補零向量 (保持索引對齊)
            # 實際應用建議在外部先過濾壞圖
            bs = imgs.shape[0]
            # predictor[-1] 是最後一層 Linear
            out_dim = model.predictor[-1].out_features
            embs_list.append(torch.zeros(bs, out_dim))
            continue

        imgs = imgs.to(device)

        # 提取特徵：Backbone -> Projector -> Normalize
        # 注意：SimSiam 論文中提到檢索可用 Projector 輸出，亦可用 Backbone 輸出。
        # 這裡為了與訓練目標一致，使用 Projector 輸出。
        f = model.backbone(imgs)
        z = model.projector(f)
        z = F.normalize(z, dim=1)  # L2 Normalize

        embs_list.append(z.cpu())

    if not embs_list:
        return torch.empty(0, model.predictor[-1].out_features)

    return torch.cat(embs_list, dim=0)


# -----------------------------------------------------------------------------
# Embed / Similarity (Modified)
# -----------------------------------------------------------------------------


@torch.no_grad()
def similarity_between_two_images(
    model: SimSiam, img1: str | Path, img2: str | Path, device: str, img_size: int = 224
) -> float:
    """計算兩張圖片的餘弦相似度。

    自動偵測模型輸入通道 (Grayscale/RGB) 並套用對應的前處理。
    """
    model.eval()

    # 1. 自動偵測模型通道配置
    # 檢查第一層卷積的輸入通道數
    in_channels = model.backbone.conv1.in_channels
    is_grayscale = in_channels == 1

    # 2. 設定對應的 Normalization 參數與轉換模式
    if is_grayscale:
        mean, std = [0.5], [0.5]
        convert_mode = "L"
    else:
        # ImageNet 預設統計值
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        convert_mode = "RGB"

    # 3. 建立推論用的 Transform
    tfm = make_inference_transform(img_size=img_size, mean=mean, std=std)

    def _load_and_process(path):
        img = Image.open(path).convert(convert_mode)
        # 增加 batch 維度 [C, H, W] -> [1, C, H, W]
        return tfm(img).unsqueeze(0).to(device)

    try:
        x1 = _load_and_process(img1)
        x2 = _load_and_process(img2)
    except Exception as e:
        print(f"Error loading images for similarity: {e}")
        return 0.0

    # 4. 提取特徵 (Backbone -> Projector -> Normalize)
    # 與 embed_images 保持一致，使用 Projector 輸出進行檢索
    f1 = model.backbone(x1)
    f2 = model.backbone(x2)

    z1 = F.normalize(model.projector(f1), dim=1)
    z2 = F.normalize(model.projector(f2), dim=1)

    # 5. 計算 Cosine Similarity
    sim = F.cosine_similarity(z1, z2, dim=1).item()
    return float(sim)


# -----------------------------------------------------------------------------
# Main Block (Modified)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # 修正重點：
    # 1. 使用 GeometryAwareTransform 替代不存在的 TransformTwice
    # 2. 針對工程圖情境，設定單通道 (Grayscale) 參數
    # 3. 修正 DataLoader 的 num_workers 設定 (Windows 下建議設為 0 避免錯誤)

    # 設定參數
    img_size = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模擬資料路徑 (請替換為實際路徑)
    data_dir = Path("data/images")
    if not data_dir.exists():
        print(f"警告：資料夾 {data_dir} 不存在，請修改路徑。")
        paths = []
    else:
        paths = [
            p
            for p in data_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"}
        ]

    if not paths:
        print("未找到圖片，程式結束。")
    else:
        print(f"找到 {len(paths)} 張圖片，開始設定訓練環境...")

        # 針對工程圖/灰階圖的設定 (mean/std 設為 0.5)
        mean, std = [0.5], [0.5]

        # 初始化幾何感知增強 (回傳兩視角)
        transform = GeometryAwareTransform(img_size=img_size, mean=mean, std=std)

        # 建立資料集 (強制轉為灰階以符合工程圖需求)
        ds = UnlabeledImages(paths, transform=transform, grayscale=True)

        dl = DataLoader(
            ds,
            batch_size=32,  # 根據顯存調整
            shuffle=True,
            drop_last=True,
            num_workers=4,  # 若在 Windows 報錯請改為 0
            pin_memory=True,
        )

        # 初始化模型 (in_channels=1 對應灰階輸入)
        model = SimSiam(backbone="resnet18", pretrained=False, in_channels=1).to(device)

        # 優化器設定
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        # 混合精度訓練
        scaler = torch.amp.GradScaler("cuda", enabled=(device.startswith("cuda")))

        print(f"開始訓練 (Device: {device})...")
        for epoch in range(1, 6):  # 演示用跑 5 個 epoch
            loss = train_one_epoch(model, dl, opt, scaler, device)
            print(f"Epoch {epoch:02d} | Loss = {loss:.4f}")

        print("訓練完成。")

        # 簡單測試相似度功能 (取列表中前兩張圖)
        if len(paths) >= 2:
            sim_score = similarity_between_two_images(model, paths[0], paths[1], device)
            print(
                f"Similarity between {paths[0].name} and {paths[1].name}: {sim_score:.4f}"
            )
