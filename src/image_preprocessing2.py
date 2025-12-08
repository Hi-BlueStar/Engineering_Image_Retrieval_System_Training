# image_preprocessingcopy.py
# -*- coding: utf-8 -*-
"""
實用工具流程：
- 從二值化影像中尋找大小組件。
- 使用外部輪廓填滿的遮罩來判斷小元件是否完全位於大元件內部。
- 將指定的小組件合併回其對應的大組件。
- 儲存：原始影像、已清理/合且位置與原始影像相同的影像，以及一個合併後的大組件在畫布中隨機重新排列且不重疊的版本。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import os
import cv2
import time
import numpy as np

import functools
import gc
import sys
import time
import tracemalloc

try:
    import psutil  # 若未安裝，將自動退化為只用 tracemalloc
except Exception:
    psutil = None

# ==============================
# 通用工具
# ==============================
def timer(func):
    """裝飾器：計算函數執行時間"""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # 高精度起始時間
        result = func(*args, **kwargs)
        end = time.perf_counter()  # 高精度結束時間
        duration = end - start
        print(f"函數 {func.__name__} 執行時間：{duration:.6f} 秒")
        return result
    return wrapper

def show_memory(label: str | None = None, *, force_gc: bool = True, stream=None):
    """
    量測函式執行的記憶體使用量並列印摘要。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            out = stream or sys.stdout
            name = label or func.__name__

            if force_gc:
                gc.collect()

            # OS 層級 RSS（若可用）
            proc = psutil.Process() if psutil is not None else None
            rss_before = proc.memory_info().rss if proc else None

            # Python 配置追蹤
            started_here = False
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                started_here = True

            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                current, peak = tracemalloc.get_traced_memory()
                if started_here:
                    tracemalloc.stop()

                rss_after = proc.memory_info().rss if proc else None
                rss_delta = (rss_after - rss_before) if (proc and rss_before is not None) else None

                def _fmt_bytes(n: int) -> str:
                    return f"{n / (1024 * 1024):.2f} MB"

                parts = []
                if rss_delta is not None:
                    parts.append(f"ΔRSS={_fmt_bytes(rss_delta)} (from {_fmt_bytes(rss_before)} to {_fmt_bytes(rss_after)})")
                else:
                    parts.append("ΔRSS=（psutil 未安裝，略過）")
                parts.append(f"Python峰值配置(peak)={_fmt_bytes(peak)}")
                parts.append(f"耗時={elapsed:.3f}s")

                print(f"[MEM] {name}: " + ", ".join(parts), file=out)

                wrapper._last_memory_report = {
                    "rss_before": rss_before,
                    "rss_after": rss_after,
                    "rss_delta": rss_delta,
                    "py_peak_bytes": peak,
                    "elapsed_sec": elapsed,
                }
        return wrapper
    return decorator

# ------------------------------------------------------------
# 共用 I/O 工具
# ------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def imwrite_unicode(path: Path, img: np.ndarray) -> bool:
    """
    儲存影像，對含非 ASCII 路徑改用 imencode + tofile 避免 Windows 編碼問題。
    """
    ensure_dir(path.parent)
    suffix = path.suffix or ".png"
    ext = suffix if suffix.startswith(".") else f".{suffix}"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False

def _save_step(name: str, img: np.ndarray, out_dir: Optional[Path]) -> None:
    if out_dir is None:
        return
    ensure_dir(out_dir)
    imwrite_unicode(out_dir / f"{name}.png", img)


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    """
    OpenCV 在 Windows 上對含非 ASCII/中文路徑常有編碼問題，改用 imdecode。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _compose_single_component(
    src: np.ndarray,
    mask01: np.ndarray,
    bg_mode: str,
) -> np.ndarray:
    """
    將單一遮罩覆蓋到與原圖同大小的畫布上。
    - 背景顏色依據 bg_mode（白底/黑底）填充。
    - 前景畫素從原圖擷取，確保保留原始細節。
    """
    bg_val = 255 if bg_mode == "white" else 0
    canvas = np.full_like(src, bg_val)
    idx = mask01.astype(bool)
    canvas[idx] = src[idx]
    return canvas


def save_large_components_images(
    src: np.ndarray,
    large: List[Component],
    merged_masks: Dict[int, np.ndarray],
    bg_mode: str,
    out_root: Path,
    padding: int = 2,
) -> List[Path]:
    """
    將「處理後的大元件（含貼回的小元件）」各自輸出成獨立影像。
    影像大小為「大元件外框 + padding」的裁切區域，保留原圖畫素，背景按底色填充。
    """
    out_dir = out_root / "large_components"
    ensure_dir(out_dir)
    saved = []
    H, W, _ = src.shape
    bg_val = 255 if bg_mode == "white" else 0
    for L in large:
        mask_full = merged_masks[L.label]
        x, y, w, h = L.bbox
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(W, x + w + padding)
        y1 = min(H, y + h + padding)

        roi_mask = mask_full[y0:y1, x0:x1]
        roi_src = src[y0:y1, x0:x1]

        canvas = np.full_like(roi_src, bg_val)
        idx = roi_mask.astype(bool)
        canvas[idx] = roi_src[idx]
        comp_img = canvas

        out_path = out_dir / f"large_L{L.label}_area{L.area}_pad{padding}.png"
        if not imwrite_unicode(out_path, comp_img):
            raise IOError(f"寫檔失敗：{out_path}")
        saved.append(out_path)
    return saved

# ------------------------------------------------------------
# 影像前處理：自動二值化（前景=1, 背景=0）
# ------------------------------------------------------------
def auto_binarize(img_bgr: np.ndarray, bin_thresh: int = 0) -> Tuple[np.ndarray, str]:
    """
    回傳：(bw01, bg_mode)；bw01 為 0/1，bg_mode ∈ {"white","black"}（原圖底色）
    - 若未指定閾值，使用 Otsu 自適應；同時計算黑/白底版本，選取前景像素數較少的一種。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if bin_thresh and 0 < bin_thresh < 255:
        _, bin_white = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY)
        _, bin_black = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bin_white = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin_black = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnt_w = int(np.count_nonzero(bin_white))
    cnt_b = int(np.count_nonzero(bin_black))
    h, w = gray.shape
    total = h * w
    candidates = [(cnt_w, "black_bg", bin_white), (cnt_b, "white_bg", bin_black)]
    candidates = [(c, tag, b) for (c, tag, b) in candidates if 0 < c < total * 0.95]
    chosen_cnt, chosen_tag, chosen = min(candidates, key=lambda t: t[0]) if candidates else (cnt_b, "white_bg", bin_black)

    bw = (chosen > 0).astype(np.uint8)  # 0/1
    bg_mode = "white" if chosen_tag == "white_bg" else "black"
    return bw, bg_mode

# ------------------------------------------------------------
# 連通元件分析
# ------------------------------------------------------------
@dataclass
class Component:
    label: int
    area: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    mask: np.ndarray  # 與原圖同大小的 0/1 uint8

def analyze_components(bw01: np.ndarray) -> List[Component]:
    """
    對二值圖(bw01: 前景=1)做連通元件；回傳 Component list 依面積遞減。
    面積定義更新為：元件的外接矩形面積 (w*h)。
    使用 8-connectivity 以避免細節被拆分。
    """
    num_labels, labels = cv2.connectedComponents(bw01, connectivity=8)
    comps: List[Component] = []
    for lbl in range(1, num_labels):
        mask = (labels == lbl).astype(np.uint8)
        pix_area = int(cv2.countNonZero(mask))
        if pix_area == 0:
            continue
        x, y, w, h = cv2.boundingRect(mask)
        bbox_area = int(w) * int(h)
        comps.append(Component(lbl, bbox_area, (x, y, w, h), mask))
    comps.sort(key=lambda c: c.area, reverse=True)
    return comps

def select_large_small(comps: List[Component], top_n: int, remove_largest: bool
                       ) -> Tuple[List[Component], List[Component]]:
    """
    回傳 (large_components, small_components)
    """
    ordered = comps.copy()
    if remove_largest and ordered:
        ordered = ordered[1:]
    large = ordered[:top_n]
    small = ordered[top_n:]
    return large, small

# ------------------------------------------------------------
# 「完全落在」判定（遮罩方式，不用 bbox）
# ------------------------------------------------------------
def filled_region_from_component(comp: Component) -> np.ndarray:
    """
    由單一大元件的 0/1 mask，取 external contour 後填滿，得到其「外輪廓內部」區域。
    回傳：uint8 0/1，大小等同原圖。
    這種方式能避免僅使用 bbox 造成的偽包含。
    """
    mask255 = (comp.mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask255)
    if contours:
        cv2.drawContours(filled, contours, -1, color=255, thickness=-1)
    return (filled > 0).astype(np.uint8)

def assign_small_to_large(large: List[Component], small: List[Component]
                         ) -> Dict[int, List[Component]]:
    """
    對每個小元件判斷是否『完全落在』某一個大元件的外輪廓填滿區域內；若是則指派給該大元件。
    回傳：{large_label: [small_components...]}
    """
    if not large or not small:
        return {L.label: [] for L in large}

    filled_map = {L.label: filled_region_from_component(L) for L in large}
    assignment: Dict[int, List[Component]] = {L.label: [] for L in large}

    for s in small:
        ys, xs = np.where(s.mask > 0)
        if len(ys) == 0:
            continue
        best_label = None
        best_cover = -1
        fully_inside_any = False

        for L in large:
            f = filled_map[L.label]
            cover_vals = f[ys, xs]
            if np.all(cover_vals == 1):
                fully_inside_any = True
                cover = int(np.sum(cover_vals))
                if cover > best_cover:
                    best_cover = cover
                    best_label = L.label

        if fully_inside_any and best_label is not None:
            assignment[best_label].append(s)

    return assignment

# ------------------------------------------------------------
# 小元件貼回：將指派給大元件的小元件遮罩與大元件遮罩合併
# ------------------------------------------------------------
def merge_small_into_large(
    large: List[Component], assignment: Dict[int, List[Component]]
) -> Dict[int, np.ndarray]:
    """
    Union each large component mask with the masks of the small components assigned to it.
    """
    merged: Dict[int, np.ndarray] = {}
    for L in large:
        merged_mask = L.mask.copy()
        for s in assignment.get(L.label, []):
            merged_mask = np.maximum(merged_mask, s.mask)
        merged[L.label] = merged_mask
    return merged


def _boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], padding: int = 0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + padding <= bx
        or bx + bw + padding <= ax
        or ay + ah + padding <= by
        or by + bh + padding <= ay
    )


def compose_on_original_positions(
    src: np.ndarray, merged_masks: Dict[int, np.ndarray], bg_mode: str
) -> np.ndarray:
    """
    Keep only merged large components (large + assigned small) at their original positions.
    Background matches the detected bg_mode.
    """
    bg_val = 255 if bg_mode == "white" else 0
    canvas = np.full_like(src, bg_val)
    for mask in merged_masks.values():
        idx = mask.astype(bool)
        canvas[idx] = src[idx]
    return canvas


def random_arrange_components(
    src: np.ndarray,
    large: List[Component],
    merged_masks: Dict[int, np.ndarray],
    bg_mode: str,
    rng: Optional[random.Random] = None,
    *,
    padding: int = 2,
    max_attempts: int = 400,
) -> np.ndarray:
    """
    Randomly place each merged large component (with its small components) on the canvas without overlap.
    - 以大元件 bbox 當作「擺放盒子」；內部畫素取自貼回後的 ROI。
    - 嘗試 max_attempts 次找不重疊位置，失敗則回退至原始位置（若仍重疊則略過）。
    """
    rng = rng or random.Random()
    H, W, _ = src.shape
    bg_val = 255 if bg_mode == "white" else 0
    canvas = np.full_like(src, bg_val)

    comps = []
    for L in large:
        x, y, w, h = L.bbox
        mask_full = merged_masks[L.label]
        roi_mask = mask_full[y : y + h, x : x + w]
        roi_src = src[y : y + h, x : x + w]
        comps.append({"bbox": (x, y, w, h), "roi_mask": roi_mask, "roi_src": roi_src})

    rng.shuffle(comps)
    placed: List[Tuple[int, int, int, int]] = []

    for comp in comps:
        w = comp["bbox"][2]
        h = comp["bbox"][3]
        chosen: Optional[Tuple[int, int, int, int]] = None

        for _ in range(max_attempts):
            rx = rng.randint(0, max(0, W - w))
            ry = rng.randint(0, max(0, H - h))
            candidate = (rx, ry, w, h)
            if not any(_boxes_overlap(candidate, p, padding) for p in placed):
                chosen = candidate
                break

        if chosen is None:
            # Fallback to original spot if it fits and does not overlap already placed boxes
            candidate = comp["bbox"]
            if (
                candidate[0] + candidate[2] <= W
                and candidate[1] + candidate[3] <= H
                and not any(_boxes_overlap(candidate, p, padding) for p in placed)
            ):
                chosen = candidate
            else:
                continue  # skip if nowhere to go

        rx, ry, w, h = chosen
        dst = canvas[ry : ry + h, rx : rx + w]
        mask_bool = comp["roi_mask"].astype(bool)
        dst[mask_bool] = comp["roi_src"][mask_bool]
        placed.append(chosen)

    return canvas


# ------------------------------------------------------------
# 主要流程（已改為：僅處理 TopK 大元件）
# ------------------------------------------------------------
@timer
@show_memory("主要流程")
def run_pipeline(
    input_path: str,
    output_dir: str,
    top_n: int = 5,
    remove_largest: bool = True,
    seed: Optional[int] = None,
    padding: int = 2,
    max_attempts: int = 400,
    random_count: int = 10,
) -> Dict[str, object]:
    """
    主要流程（無 CLI，直接呼叫此函式）
    - Step 1: 連通元件分析、排序，劃分大/小元件（TopK，大元件可選擇移除最大一個）
    - Step 2: 以「外輪廓填滿遮罩」判斷小元件是否完整落在某大元件內並指派
    - Step 3: 將指派的小元件貼回大元件，輸出：
        * 原圖
        * 大元件貼回後、保持原位置的全圖
        * 大元件貼回後隨機排列且不重疊的全圖
        * 每個「處理後大元件」的獨立輸出圖（裁切為外框+padding，存放於 large_components/）

    Args:
        input_path: 輸入影像路徑
        output_dir: 輸出目錄
        top_n: 要保留的大元件數
        remove_largest: 是否先移除面積最大元件再取 TopK
        seed: 隨機排列的亂數種子（固定可重現）
        padding: 隨機排列時盒子之間預留的間距（像素），同時也作為大元件裁切輸出的外框 padding
        max_attempts: 為每個元件嘗試隨機放置的最大迭代次數
        random_count: 要產生的「隨機排列圖」張數（預設 10 張）

    Returns:
        Dict[str, object]: 各輸出檔案的路徑
            - "original": Path
            - "merged": Path
            - "random": List[Path]（多張隨機排列圖）
            - "large_dir": Path（存放單獨大元件）
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到影像檔：{input_path}")

    output_root = Path(output_dir)
    ensure_dir(output_root)
    rng = random.Random(seed)

    # 讀檔 + 自動二值化（支援白/黑底）
    src = _imread_unicode(input_path)
    if src is None:
        raise ValueError(f"cv2.imread/imdecode 讀取失敗：{input_path.resolve()}")
    bw01, bg_mode = auto_binarize(src)  # 前景=1，bg_mode={"white","black"}

    # 連通元件、排序、定義大/小
    comps = analyze_components(bw01)
    large, small = select_large_small(comps, top_n, remove_largest)
    
    # 以填滿外輪廓的遮罩決定小元件是否完全包含於大元件，並完成指派
    assignment = assign_small_to_large(large, small)  # 採用填充式外輪廓
    # 合併：將小元件貼回對應的大元件遮罩，得到「處理後大元件」
    merged_masks = merge_small_into_large(large, assignment)

    # 生成「大元件貼回後、保持原位置」的全圖
    merged_img = compose_on_original_positions(src, merged_masks, bg_mode)
    # 生成「大元件貼回後、隨機排列且不重疊」的多張全圖
    random_imgs: List[Tuple[np.ndarray, int]] = []
    for idx_random in range(random_count):
        random_img = random_arrange_components(
            src,
            large,
            merged_masks,
            bg_mode,
            rng,
            padding=padding,
            max_attempts=max_attempts,
        )
        random_imgs.append((random_img, idx_random))
    # 逐一輸出「處理後大元件」影像（與原圖同尺寸，便於檢視）
    saved_large_paths = save_large_components_images(
        src,
        large,
        merged_masks,
        bg_mode,
        output_root,
        padding=padding,
    )

    base = Path(input_path).stem
    ext = Path(input_path).suffix or ".png"

    paths = {
        "original": output_root / f"{base}_original{ext}",
        "merged": output_root / f"{base}_merged{ext}",
        "random": [],
        "large_dir": output_root / "large_components",
    }

    if not imwrite_unicode(paths["original"], src):
        raise IOError(f"寫檔失敗：{paths['original']}")
    if not imwrite_unicode(paths["merged"], merged_img):
        raise IOError(f"寫檔失敗：{paths['merged']}")
    for img, idx_random in random_imgs:
        out_path = output_root / f"{base}_random_{idx_random+1:02d}{ext}"
        if not imwrite_unicode(out_path, img):
            raise IOError(f"寫檔失敗：{out_path}")
        paths["random"].append(out_path)

    print(f"[OK] 已將原始文件儲存到 {paths['original']}")
    print(f"[OK] 已儲存合併（包含內部小文件） {paths['merged']}")
    print(f"[OK] 已儲存隨機排列 {len(paths['random'])} 張；範例：{paths['random'][0] if paths['random'] else '無'}")
    print(f"[OK] 已輸出處理後大元件於資料夾：{paths['large_dir']} ({len(saved_large_paths)} files)")

    return paths

# ------------------------------------------------------------
# 範例（供引用端參考）
# ------------------------------------------------------------
if __name__ == "__main__":
    # 注意：本模組不提供 CLI；以下段落僅作為「在 IDE 中直接執行」的測試入口。
    
    # --- 範例 1：完整輸出模式 ---
    demo_path = Path("data/engineering_images_100dpi/固定夾塊/175H10-DSV-000-10200.png")
    if not demo_path.is_file():
        print(f"[WARN] 範例檔案不存在：{demo_path.resolve()}")
    else:
        run_pipeline(
            str(demo_path),
            output_dir="results/random_arrange_demo",
            top_n=5,
            remove_largest=True,
            seed=420,
            padding=4,          # 加大間距，避免隨機排列時貼太近
            max_attempts=800,   # 增加嘗試次數，提高放置成功率
            random_count=10,    # 產生 10 張隨機排列結果
        )
