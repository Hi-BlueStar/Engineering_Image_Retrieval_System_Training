# image_preprocessingcopy.py
"""
實用工具流程：
- 從二值化影像中尋找大小組件。
- 使用外部輪廓填滿的遮罩來判斷小元件是否完全位於大元件內部。
- 將指定的小組件合併回其對應的大組件。
- 儲存：原始影像、已清理/合且位置與原始影像相同的影像，以及一個合併後的大組件在畫布中隨機重新排列且不重疊的版本。
"""

from __future__ import annotations

import functools
import gc
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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
                rss_delta = (
                    (rss_after - rss_before)
                    if (proc and rss_before is not None)
                    else None
                )

                def _fmt_bytes(n: int) -> str:
                    return f"{n / (1024 * 1024):.2f} MB"

                parts = []
                if rss_delta is not None:
                    parts.append(
                        f"ΔRSS={_fmt_bytes(rss_delta)} (from {_fmt_bytes(rss_before)} to {_fmt_bytes(rss_after)})"
                    )
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
    優化：為 PNG 格式設置較低的壓縮率以加快寫檔速度。
    """
    ensure_dir(path.parent)
    suffix = path.suffix or ".png"
    ext = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    
    # 設置更快的 PNG 壓縮級別 (1) 以減少寫檔時間
    params = []
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    elif ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    try:
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def _save_step(name: str, img: np.ndarray, out_dir: Path | None) -> None:
    if out_dir is None:
        return
    ensure_dir(out_dir)
    imwrite_unicode(out_dir / f"{name}.png", img)


def _imread_unicode(path: Path) -> np.ndarray | None:
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


def compose_merged_image(
    src: np.ndarray,
    large: list[Component],
    merged_masks: dict[int, np.ndarray],
    bg_mode: str,
) -> np.ndarray:
    """
    (重構替代原 compose_on_original_positions)
    使用 ROI 遮罩還原全圖。
    """
    bg_val = 255 if bg_mode == "white" else 0
    canvas = np.full_like(src, bg_val)

    for L in large:
        roi_mask = merged_masks[L.label]
        x, y, w, h = L.bbox

        # 從原圖取對應位置的 pixels
        roi_src = src[y : y + h, x : x + w]

        # 貼到畫布
        roi_canvas = canvas[y : y + h, x : x + w]
        idx = roi_mask.astype(bool)
        roi_canvas[idx] = roi_src[idx]

    return canvas


def save_large_components_images(
    src: np.ndarray,
    large: list[Component],
    merged_masks: dict[int, np.ndarray],
    bg_mode: str,
    out_root: Path,
    padding: int = 2,
) -> list[Path]:
    """
    將「處理後的大元件（含貼回的小元件）」各自輸出成獨立影像。
    影像大小為「大元件外框 + padding」的裁切區域，保留原圖畫素，背景按底色填充。
    使用 ROI Mask 輸出單一大元件影像。
    """
    out_dir = out_root / "large_components"
    ensure_dir(out_dir)
    saved = []
    H, W, _ = src.shape
    bg_val = 255 if bg_mode == "white" else 0

    for L in large:
        roi_mask = merged_masks[L.label]  # 這是 ROI 大小的 mask
        x, y, w, h = L.bbox

        # 計算含 padding 的 ROI
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(W, x + w + padding)
        y1 = min(H, y + h + padding)

        # 建立一個 padding 後大小的畫布
        canvas_h, canvas_w = y1 - y0, x1 - x0
        comp_img = np.full((canvas_h, canvas_w, 3), bg_val, dtype=np.uint8)

        # 計算 roi_mask 在新畫布中的偏移位置
        # mask 的左上角是 (x, y)，畫布左上角是 (x0, y0)
        off_x = x - x0
        off_y = y - y0

        # 來源像素
        roi_src = src[y : y + h, x : x + w]

        # 透過 mask 決定要複製哪些像素
        idx = roi_mask.astype(bool)

        # 目標區域
        target_area = comp_img[off_y : off_y + h, off_x : off_x + w]
        target_area[idx] = roi_src[idx]

        out_path = out_dir / f"large_L{L.label}_area{L.area}_pad{padding}.png"
        if not imwrite_unicode(out_path, comp_img):
            raise OSError(f"寫檔失敗：{out_path}")
        saved.append(out_path)
    return saved


# ------------------------------------------------------------
# 影像前處理：自動二值化（前景=1, 背景=0）
# ------------------------------------------------------------
def auto_binarize(img_bgr: np.ndarray, bin_thresh: int = 0) -> tuple[np.ndarray, str]:
    """
    回傳：(bw01, bg_mode)；bw01 為 0/1，bg_mode ∈ {"white","black"}（原圖底色）
    - 若未指定閾值，使用 Otsu 自適應；同時計算黑/白底版本，選取前景像素數較少的一種。
    優化：僅執行一次閾值化以減少運算，利用快速的 cv2.countNonZero 與位元右移運算。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if bin_thresh and 0 < bin_thresh < 255:
        _, bin_white = cv2.threshold(gray, bin_thresh, 255, cv2.THRESH_BINARY)
    else:
        _, bin_white = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 快速計算非零像素數 (cv2.countNonZero 比 np.count_nonzero 快)
    cnt_w = cv2.countNonZero(bin_white)
    total = gray.size
    cnt_b = total - cnt_w
    
    candidates = [(cnt_w, "black_bg"), (cnt_b, "white_bg")]
    candidates = [c for c in candidates if 0 < c[0] < total * 0.95]
    chosen_cnt, chosen_tag = (
        min(candidates, key=lambda t: t[0])
        if candidates
        else (cnt_b, "white_bg")
    )

    if chosen_tag == "white_bg":
        chosen = cv2.bitwise_not(bin_white)
        bg_mode = "white"
    else:
        chosen = bin_white
        bg_mode = "black"

    # 位元右移 7 位將 0/255 快速轉換為 0/1 (比 .astype(np.uint8) 快)
    bw = chosen >> 7
    return bw, bg_mode


# ------------------------------------------------------------
# 連通元件分析
# ------------------------------------------------------------
@dataclass
class Component:
    label: int
    area: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    roi_mask: np.ndarray | None = None  # 優化：延遲載入/選用的 ROI 遮罩，減少無謂記憶體與 CPU 開銷


def analyze_components(bw01: np.ndarray, return_labels: bool = False) -> list[Component] | tuple[list[Component], np.ndarray]:
    """
    對二值圖(bw01: 前景=1)做連通元件；回傳 Component 列表（roi_mask 初始化為 None）及可選的 labels 標記圖。
    優化：支持 return_labels 以確保相容性，且使用列表推導式加快 Component 實例化。
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw01, connectivity=8
    )

    # 快速列表推導式
    comps = [
        Component(lbl, stats[lbl, 2] * stats[lbl, 3], tuple(stats[lbl, :4]), None)
        for lbl in range(1, num_labels)
    ]
    comps.sort(key=lambda c: c.area, reverse=True)
    
    if return_labels:
        return comps, labels
    return comps


def select_large_small(
    comps: list[Component], top_n: int, remove_largest: bool
) -> tuple[list[Component], list[Component]]:
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
    優化：僅在 ROI 範圍內進行輪廓查找與繪製。
    """
    mask255 = (comp.roi_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 建立 ROI 大小的 filled mask
    filled_roi = np.zeros_like(comp.roi_mask)
    if contours:
        cv2.drawContours(filled_roi, contours, -1, color=1, thickness=-1)

    return filled_roi


def assign_small_to_large(
    large: list[Component], small: list[Component], labels: np.ndarray | None = None
) -> dict[int, list[Component]]:
    """
    對每個小元件判斷是否『完全落在』某一個大元件的外輪廓填滿區域內；若是則指派給該大元件。
    回傳：{large_label: [small_components...]}
    優化：使用 numpy 矩陣切片與標籤值直接判斷，避免為所有小元件提取 roi_mask，也免除對大區域進行 np.unique 排序，顯著提速。
    """
    if not large or not small:
        return {L.label: [] for L in large}

    filled_map = {L.label: filled_region_from_component(L) for L in large}
    assignment: dict[int, list[Component]] = {L.label: [] for L in large}

    for s in small:
        sx, sy, sw, sh = s.bbox
        for L in large:
            Lx, Ly, Lw, Lh = L.bbox

            # --- BBox 快速排除 ---
            if not (
                Lx <= sx
                and Ly <= sy
                and (Lx + Lw) >= (sx + sw)
                and (Ly + Lh) >= (sy + sh)
            ):
                continue

            offset_y = sy - Ly
            offset_x = sx - Lx

            f_roi = filled_map[L.label]
            f_roi_slice = f_roi[offset_y : offset_y + sh, offset_x : offset_x + sw]

            if labels is not None:
                # 向量化切片判定：檢查在小元件所在的區域內，對應 labels 值為其 label 的像素在 f_roi 中是否全為 1
                labels_slice = labels[sy : sy + sh, sx : sx + sw]
                if np.all(f_roi_slice[labels_slice == s.label]):
                    assignment[L.label].append(s)
                    break
            else:
                # 退化相容判定
                if s.roi_mask is None:
                    continue
                if np.all(s.roi_mask <= f_roi_slice):
                    assignment[L.label].append(s)
                    break

    return assignment


# ------------------------------------------------------------
# 小元件貼回：將指派給大元件的小元件遮罩與大元件遮罩合併
# ------------------------------------------------------------
def merge_small_into_large(
    large: list[Component], assignment: dict[int, list[Component]], labels: np.ndarray | None = None
) -> dict[int, np.ndarray]:
    """
    Union each large component mask with the masks of the small components assigned to it.
    優化：當 s.roi_mask 為 None 且提供了 labels 時，才延遲載入小元件遮罩，以節省記憶體。
    """
    merged: dict[int, np.ndarray] = {}
    for L in large:
        Lx, Ly, Lw, Lh = L.bbox
        merged_mask = L.roi_mask.copy()

        for s in assignment.get(L.label, []):
            sx, sy, sw, sh = s.bbox
            # 計算小元件在大元件 ROI 中的偏移量
            offset_x = sx - Lx
            offset_y = sy - Ly

            if s.roi_mask is not None:
                s_mask = s.roi_mask
            elif labels is not None:
                s_labels = labels[sy : sy + sh, sx : sx + sw]
                s_mask = (s_labels == s.label).astype(np.uint8)
            else:
                continue

            # 使用 np.maximum 避免覆蓋掉原本存在的 mask
            roi_slice = merged_mask[offset_y : offset_y + sh, offset_x : offset_x + sw]
            merged_mask[offset_y : offset_y + sh, offset_x : offset_x + sw] = (
                np.maximum(roi_slice, s_mask)
            )

        merged[L.label] = merged_mask
    return merged


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 0
) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + padding <= bx
        or bx + bw + padding <= ax
        or ay + ah + padding <= by
        or by + bh + padding <= ay
    )


def compose_on_original_positions(
    src: np.ndarray, merged_masks: dict[int, np.ndarray], bg_mode: str
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
    large: list[Component],
    merged_masks: dict[int, np.ndarray],
    bg_mode: str,
    rng: random.Random | None = None,
    *,
    padding: int = 2,
    max_attempts: int = 400,
) -> np.ndarray:
    """
    Randomly place each merged large component (with its small components) on the canvas without overlap.
    - 以大元件 bbox 當作「擺放盒子」；內部畫素取自貼回後的 ROI。
    - 嘗試 max_attempts 次找不重疊位置，失敗則回退至原始位置（若仍重疊則略過）。
    優化：預先轉換 mask 為 bool 類型以加快 np.copyto 處理。
    """
    rng = rng or random.Random()
    H, W, _ = src.shape
    bg_val = 255 if bg_mode == "white" else 0
    canvas = np.full_like(src, bg_val)

    comps_data = []
    for L in large:
        x, y, w, h = L.bbox
        roi_mask = merged_masks[L.label]
        # 預先轉換為 bool 以免在 np.copyto 中重覆運算，並增加維度
        roi_mask_bool = roi_mask.astype(bool)[:, :, None]
        roi_src = src[y : y + h, x : x + w]
        comps_data.append(
            {"bbox": (x, y, w, h), "roi_mask": roi_mask_bool, "roi_src": roi_src}
        )

    rng.shuffle(comps_data)
    placed: list[tuple[int, int, int, int]] = []

    for comp in comps_data:
        w = comp["bbox"][2]
        h = comp["bbox"][3]
        chosen = None

        for _ in range(max_attempts):
            rx = rng.randint(0, max(0, W - w))
            ry = rng.randint(0, max(0, H - h))
            candidate = (rx, ry, w, h)
            if not any(_boxes_overlap(candidate, p, padding) for p in placed):
                chosen = candidate
                break

        # Fallback logic
        if chosen is None:
            candidate = comp["bbox"]
            if (
                candidate[0] + candidate[2] <= W
                and candidate[1] + candidate[3] <= H
                and not any(_boxes_overlap(candidate, p, padding) for p in placed)
            ):
                chosen = candidate
            else:
                continue

        rx, ry, w, h = chosen
        dst = canvas[ry : ry + h, rx : rx + w]
        # 使用快速的 C-level np.copyto
        np.copyto(dst, comp["roi_src"], where=comp["roi_mask"])
        placed.append(chosen)

    return canvas


# ------------------------------------------------------------
# 主要流程
# ------------------------------------------------------------
@timer
@show_memory("主要流程")
def run_pipeline(
    input_path: str,
    output_dir: str,
    top_n: int = 5,
    remove_largest: bool = True,
    seed: int | None = None,
    padding: int = 2,
    max_attempts: int = 400,
    random_count: int = 10,
) -> dict[str, object]:
    """
    主要流程
    優化：完全重構為極速向量化版本。
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到影像檔：{input_path}")

    output_root = Path(output_dir)
    ensure_dir(output_root)
    rng = random.Random(seed)

    src = _imread_unicode(input_path)
    if src is None:
        raise ValueError(f"cv2.imread/imdecode 讀取失敗：{input_path.resolve()}")
    bw01, bg_mode = auto_binarize(src)  # 前景=1，bg_mode={"white","black"}

    # 1. 向量化獲取連通元件與標記
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bw01, connectivity=8
    )

    # 2. 向量化排序 (依 BBox 面積排序)
    areas = stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]
    sorted_idx = np.argsort(areas)[::-1] + 1  # 1-based labels

    if remove_largest and len(sorted_idx) > 0:
        sorted_idx = sorted_idx[1:]

    large_labels = sorted_idx[:top_n]
    small_labels = sorted_idx[top_n:]

    # 3. 僅為大元件實例化 Component
    large = []
    for lbl in large_labels:
        x, y, w, h, _ = stats[lbl]
        large.append(Component(lbl, w * h, (x, y, w, h), None))

    # 4. 生成大元件遮罩
    for L in large:
        Lx, Ly, Lw, Lh = L.bbox
        roi_labels = labels[Ly : Ly + Lh, Lx : Lx + Lw]
        L.roi_mask = (roi_labels == L.label).astype(np.uint8)

    # 5. 向量化預篩選小元件：只有當它的 bbox 完全落在某個大元件的 bbox 內時才保留
    s_stats = stats[small_labels]
    s_x, s_y, s_w, s_h = s_stats[:, 0], s_stats[:, 1], s_stats[:, 2], s_stats[:, 3]
    contained_mask = np.zeros(len(small_labels), dtype=bool)
    for L in large:
        Lx, Ly, Lw, Lh = L.bbox
        mask = (s_x >= Lx) & (s_y >= Ly) & ((s_x + s_w) <= (Lx + Lw)) & ((s_y + s_h) <= (Ly + Lh))
        contained_mask |= mask

    filtered_small_labels = small_labels[contained_mask]
    
    # 僅為通過篩選的小元件實例化 Component (不預提取 roi_mask)
    filtered_small = []
    for lbl in filtered_small_labels:
        x, y, w, h, _ = stats[lbl]
        filtered_small.append(Component(lbl, w * h, (x, y, w, h), None))

    # 6. 極速判定包含關係
    assignment = assign_small_to_large(large, filtered_small, labels)
    
    # 7. 合併：僅在需要時提取小元件遮罩
    merged_masks = merge_small_into_large(large, assignment, labels)

    # 生成「大元件貼回後、保持原位置」的全圖
    merged_img = compose_merged_image(src, large, merged_masks, bg_mode)
    # 生成「大元件貼回後、隨機排列且不重疊」的多張全圖
    random_imgs: list[tuple[np.ndarray, int]] = []
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

    # 逐一輸出「處理後大元件」影像
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
        raise OSError(f"寫檔失敗：{paths['original']}")
    if not imwrite_unicode(paths["merged"], merged_img):
        raise OSError(f"寫檔失敗：{paths['merged']}")
    for img, idx_random in random_imgs:
        out_path = output_root / f"{base}_random_{idx_random + 1:02d}{ext}"
        if not imwrite_unicode(out_path, img):
            raise OSError(f"寫檔失敗：{out_path}")
        paths["random"].append(out_path)

    print(f"[OK] 已將原始文件儲存到 {paths['original']}")
    print(f"[OK] 已儲存合併（包含內部小文件） {paths['merged']}")
    print(
        f"[OK] 已儲存隨機排列 {len(paths['random'])} 張；範例：{paths['random'][0] if paths['random'] else '無'}"
    )
    print(
        f"[OK] 已輸出處理後大元件於資料夾：{paths['large_dir']} ({len(saved_large_paths)} files)"
    )

    return paths


# ------------------------------------------------------------
# 範例
# ------------------------------------------------------------
if __name__ == "__main__":
    demo_path = Path("data/engineering_images_100dpi/175H10-DSV-000-10200.png")
    if not demo_path.is_file():
        print(f"[WARN] 範例檔案不存在：{demo_path.resolve()}")
    else:
        run_pipeline(
            str(demo_path),
            output_dir="results_test/random_arrange_demo",
            top_n=5,
            remove_largest=True,
            seed=420,
            padding=4,
            max_attempts=800,
            random_count=10,
        )
