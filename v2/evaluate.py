"""獨立評估管線入口 (Standalone Evaluation Pipeline Entry Point)。

============================================================
Pipeline: 載入 Checkpoint → 建立有標籤資料集 → 計算檢索指標
          → （可選）比較原始轉檔影像 vs 前處理後影像

特性：
    - 無 Data Augmentation（LabeledImageDataset 僅做 Letterbox + Normalize）
    - 以 config 中的 eval.checkpoint_path 載入訓練好的權重
    - 支援 torch.compile() 存檔（自動移除 _orig_mod. prefix）
    - 設定 eval.preprocessed_labeled_data_path 即啟用 raw vs preprocessed 比較

使用方式::

    python v2/evaluate.py --config v2/configs/eval.yaml
    python v2/evaluate.py --config v2/configs/eval.yaml \\
        eval.checkpoint_path=outputs/exp/checkpoints/checkpoint_best.pth
    python v2/evaluate.py --config v2/configs/eval.yaml \\
        eval.preprocessed_labeled_data_path=data/preprocessed_labeled_images
============================================================
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --- 確保專案根目錄在 Python Path ---
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.dataset.labeled_dataset import LabeledImageDataset
from src.evaluation.evaluator import evaluate_model, save_metrics
from src.logger import get_logger, setup_logging
from src.model.simsiam import SimSiam

logger = get_logger(__name__)


# ============================================================
# Helper functions
# ============================================================


def _strip_compile_prefix(state_dict: dict) -> "OrderedDict[str, torch.Tensor]":
    """移除 torch.compile() 在 state_dict key 前加的 '_orig_mod.' prefix。

    若 checkpoint 是從 torch.compile() 包裝的模型存下，所有 key 都會帶有此 prefix，
    直接 load_state_dict 到普通 SimSiam 會因 key 不符而失敗。
    """
    prefix = "_orig_mod."
    new_sd: OrderedDict[str, torch.Tensor] = OrderedDict()
    for k, v in state_dict.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new_sd[new_key] = v
    return new_sd


def _load_model(cfg: AppConfig) -> Tuple[SimSiam, str, dict]:
    """建構 SimSiam 模型並從 checkpoint 載入權重。

    Returns:
        (model, device, ckpt_meta) — 已 eval() 的模型、裝置字串、checkpoint 元資料。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = cfg.model
    ev = cfg.eval

    ckpt_path = Path(ev.checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

    model = SimSiam(
        backbone=m.backbone,
        proj_dim=m.proj_dim,
        proj_hidden=m.proj_hidden,
        pred_hidden=m.pred_hidden,
        pretrained=False,
        in_channels=m.in_channels,
    ).to(device)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    clean_sd = _strip_compile_prefix(state["state_dict"])
    model.load_state_dict(clean_sd)
    model.eval()

    ckpt_meta = {
        "checkpoint_path": str(ckpt_path),
        "epoch": state.get("epoch", -1),
        "val_loss": state.get("val_loss", None),
    }
    val_loss_str = (
        f"{ckpt_meta['val_loss']:.4f}"
        if ckpt_meta["val_loss"] is not None
        else "N/A"
    )
    logger.info(
        "Checkpoint 載入完成: %s (epoch=%d, val_loss=%s)",
        ckpt_path.name,
        ckpt_meta["epoch"],
        val_loss_str,
    )
    return model, device, ckpt_meta


def _build_dataset(root: str, cfg: AppConfig) -> LabeledImageDataset:
    """建立 LabeledImageDataset（無任何 Data Augmentation）。"""
    return LabeledImageDataset(
        root=Path(root),
        img_size=cfg.training.img_size,
        img_exts=cfg.training.img_exts,
        in_channels=cfg.model.in_channels,
    )


def _run_evaluation(
    model: SimSiam,
    dataset: LabeledImageDataset,
    cfg: AppConfig,
    device: str,
    label: str,
    output_csv_path: Optional[str] = None,
) -> Dict[str, float]:
    """執行特徵提取並計算檢索指標。"""
    ev = cfg.eval
    logger.info("評估開始 [%s]: %d 張影像", label, len(dataset))
    metrics = evaluate_model(
        model=model,
        labeled_dataset=dataset,
        device=device,
        top_k_values=ev.top_k_values,
        batch_size=ev.batch_size,
        num_workers=ev.num_workers,
        output_csv_path=output_csv_path,
    )
    return metrics


def _print_comparison_table(
    raw_metrics: Dict[str, float],
    pre_metrics: Dict[str, float],
    top_k_values: List[int],
) -> None:
    """並列印出 raw vs preprocessed 的指標比較表。"""
    METRIC_W = 28
    COL_W = 20

    header = (
        f"{'Metric':<{METRIC_W}}"
        f"{'Raw (no preprocess)':>{COL_W}}"
        f"{'Preprocessed':>{COL_W}}"
        f"{'Delta':>{COL_W}}"
    )
    sep = "-" * len(header)
    border = "=" * len(header)

    def row(name: str, key: str) -> None:
        raw_v = raw_metrics.get(key, float("nan"))
        pre_v = pre_metrics.get(key, float("nan"))
        delta = pre_v - raw_v
        sign = "+" if delta >= 0 else ""
        print(
            f"{name:<{METRIC_W}}"
            f"{raw_v:>{COL_W}.4f}"
            f"{pre_v:>{COL_W}.4f}"
            f"{sign}{delta:>{COL_W - 1}.4f}"
        )

    print()
    print(border)
    print(" Evaluation Comparison: Raw vs Preprocessed")
    print(border)
    print(header)
    print(sep)
    row("IACS (intra-class sim)", "IACS")
    row("Inter-class sim", "inter_class_avg_sim")
    row("Contrastive Margin", "contrastive_margin")
    print(sep)
    for k in top_k_values:
        row(f"Precision@{k}", f"top{k}_precision")
    print(border)
    print()


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimSiam 工程圖檢索 — 獨立評估管線",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "CLI 覆寫範例 (OmegaConf dotlist):\n"
            "  eval.checkpoint_path=outputs/exp/checkpoints/checkpoint_best.pth\n"
            "  eval.preprocessed_labeled_data_path=data/preprocessed_labeled_images\n"
            "  eval.top_k_values=[1,5,10]\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="評估設定檔路徑，例如 v2/configs/eval.yaml",
    )
    args, overrides = parser.parse_known_args()

    cfg = AppConfig.from_yaml(args.config, cli_overrides=overrides)
    ev = cfg.eval

    # --- 自動推導輸出路徑 (依據 Checkpoint 的時間戳記) ---
    import re
    # 搜尋模式: simsiam_exp_YYYYMMDD_HHMMSS
    exp_match = re.search(r"simsiam_exp_(\d{8}_\d{6})", ev.checkpoint_path)
    if exp_match:
        timestamp = exp_match.group(1)
        new_output_dir = Path("outputs") / f"evaluate_exp_{timestamp}"
        new_output_dir.mkdir(parents=True, exist_ok=True)
        # 更新 output_path 為該目錄下的 JSON
        ev.output_path = str(new_output_dir / "eval_results.json")
        logger.info("自動推導評估輸出目錄: %s", new_output_dir)

    # --- Logging ---
    log_file: Optional[str] = None
    if cfg.logging.log_to_file:
        log_file = str(Path(ev.output_path).parent / "eval.log")
    setup_logging(
        level=cfg.logging.level,
        log_file=log_file,
        use_rich=cfg.logging.use_rich,
    )

    # --- 驗證必要欄位 ---
    if not ev.checkpoint_path:
        raise ValueError(
            "eval.checkpoint_path 不可為空；"
            "請在 YAML 或 CLI 中指定 checkpoint 路徑，"
            "例如：eval.checkpoint_path=outputs/exp/checkpoints/checkpoint_best.pth"
        )
    if not ev.labeled_data_path:
        raise ValueError("eval.labeled_data_path 不可為空")

    # --- 載入模型（只載入一次，兩次評估共用） ---
    model, device, ckpt_meta = _load_model(cfg)

    # --- 評估原始轉檔影像（無前處理） ---
    raw_ds = _build_dataset(ev.labeled_data_path, cfg)
    raw_csv_path = str(Path(ev.output_path).with_name("eval_results_raw.csv"))
    raw_metrics = _run_evaluation(model, raw_ds, cfg, device, label="raw", output_csv_path=raw_csv_path)

    # --- 評估前處理後影像（可選），並輸出比較表 ---
    pre_metrics: Optional[Dict[str, float]] = None
    pre_ds: Optional[LabeledImageDataset] = None
    if ev.preprocessed_labeled_data_path:
        pre_ds = _build_dataset(ev.preprocessed_labeled_data_path, cfg)
        pre_csv_path = str(Path(ev.output_path).with_name("eval_results_preprocessed.csv"))
        pre_metrics = _run_evaluation(
            model, pre_ds, cfg, device, label="preprocessed", output_csv_path=pre_csv_path
        )
        _print_comparison_table(raw_metrics, pre_metrics, ev.top_k_values)

    # --- 組裝 JSON payload ---
    output_payload: dict = {
        "checkpoint": ckpt_meta,
        "config": {
            "model": {
                "backbone": cfg.model.backbone,
                "in_channels": cfg.model.in_channels,
                "proj_dim": cfg.model.proj_dim,
            },
            "img_size": cfg.training.img_size,
            "top_k_values": ev.top_k_values,
        },
        "raw": {
            "data_path": ev.labeled_data_path,
            "n_samples": len(raw_ds),
            "metrics": raw_metrics,
        },
    }
    if pre_metrics is not None and pre_ds is not None:
        output_payload["preprocessed"] = {
            "data_path": ev.preprocessed_labeled_data_path,
            "n_samples": len(pre_ds),
            "metrics": pre_metrics,
        }

    # --- 儲存 JSON ---
    save_metrics(
        metrics={},
        output_path=Path(ev.output_path),
        extra_info=output_payload,
    )
    logger.info("評估完成。結果已儲存至: %s", ev.output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("使用者中斷評估。")
    except Exception:
        logger.exception("評估管線異常終止")
        sys.exit(1)
