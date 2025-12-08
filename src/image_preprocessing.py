# image_preprocessing.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os
import cv2
import time
import numpy as np
from skimage.morphology import skeletonize

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


# ==============================
# 高階參數（可由呼叫端覆寫）
# ==============================
DEFAULTS = dict(
    TOP_N=5,
    REMOVE_LARGEST=True,
    ITERATIONS=10,  # 「骨架化+拓撲分析+分支修剪」的重複次數
    OUTPUT_DIR="results",
    SAVE_STEPS=True,  # 是否輸出每回合中間結果
)

# ------------------------------------------------------------
# 共用 I/O 工具
# ------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _save_step(name: str, img: np.ndarray, out_dir: Optional[Path]) -> None:
    if out_dir is None:
        return
    ensure_dir(out_dir)
    cv2.imwrite(str(out_dir / f"{name}.png"), img)

# ------------------------------------------------------------
# 影像前處理：自動二值化（前景=1, 背景=0）
# ------------------------------------------------------------
def auto_binarize(img_bgr: np.ndarray, bin_thresh: int = 0) -> Tuple[np.ndarray, str]:
    """
    回傳：(bw01, bg_mode)；bw01 為 0/1，bg_mode ∈ {"white","black"}（原圖底色）
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
# 拓撲分析與骨架處理（陣列版）
# ------------------------------------------------------------
class TopologyAnalyzerArray:
    """
    陣列版拓撲分析器：
    - 直接吃二值圖（前景=1），內部自轉為 0/255 uint8
    - 提供：骨架化、節點分類、橋段分離、分支修剪
    - 可重複迭代以加強修剪效果
    """
    def __init__(self, out_dir: Optional[Path] = None):
        self.out_dir = out_dir
        self._img_bin: Optional[np.ndarray] = None  # 0/255
        self._img_skel: Optional[np.ndarray] = None  # 0/255
        self.graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.endpoints: List[Tuple[int, int]] = []
        self.branch_points: List[Tuple[int, int]] = []
        self.bridges: List[List[Tuple[int, int]]] = []
        self.loops: List[List[Tuple[int, int]]] = []

    @staticmethod
    def _neighbors8(y: int, x: int, img_bool: np.ndarray) -> List[Tuple[int, int]]:
        h, w = img_bool.shape
        coords = []
        for j in range(y - 1, y + 2):
            for i in range(x - 1, x + 2):
                if (j, i) == (y, x):
                    continue
                if 0 <= j < h and 0 <= i < w and img_bool[j, i]:
                    coords.append((j, i))
        return coords

    def _from_binary01(self, bin01: np.ndarray) -> None:
        self._img_bin = (bin01.astype(np.uint8) * 255)
        _save_step("1_binary", self._img_bin, self.out_dir)

    def _skeletonize(self) -> None:
        bool_img = (self._img_bin > 0)
        skel_bool = skeletonize(bool_img)
        self._img_skel = (skel_bool.astype(np.uint8)) * 255
        _save_step("2_skeleton", self._img_skel, self.out_dir)

    def _build_graph(self) -> None:
        skel = (self._img_skel > 0)
        ys, xs = np.nonzero(skel)
        graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for y, x in zip(ys, xs):
            graph[(y, x)] = self._neighbors8(y, x, skel)
        self.graph = graph

    def _classify_nodes(self) -> None:
        eps, brs = [], []
        for node, nbrs in self.graph.items():
            deg = len(nbrs)
            if deg == 1:
                eps.append(node)
            elif deg > 2:
                brs.append(node)
        self.endpoints = eps
        self.branch_points = brs

        vis = cv2.cvtColor(self._img_skel, cv2.COLOR_GRAY2BGR)
        for y, x in eps:
            cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        for y, x in brs:
            cv2.circle(vis, (x, y), 2, (0, 255, 255), -1)
        _save_step("3_nodes", vis, self.out_dir)

    def _separate_components(self) -> None:
        visited = set()
        self.bridges, self.loops = [], []

        def walk_from_endpoint(ep):
            path = [ep]
            cur, prev = ep, None
            while True:
                nbrs = [n for n in self.graph[cur] if n != prev]
                visited.add(cur)
                if len(nbrs) != 1:
                    break
                prev, cur = cur, nbrs[0]
                path.append(cur)
            return path

        for ep in self.endpoints:
            if ep not in visited:
                self.bridges.append(walk_from_endpoint(ep))

        unvisited = [n for n in self.graph if n not in visited]
        core_set = []
        while unvisited:
            comp = set()
            stack = [unvisited.pop()]
            degs = []
            while stack:
                v = stack.pop()
                if v in comp:
                    continue
                comp.add(v)
                degs.append(len(self.graph[v]))
                for nb in self.graph[v]:
                    if nb not in visited and nb not in comp:
                        stack.append(nb)
            if all(d == 2 for d in degs):
                self.loops.append(list(comp))
            else:
                core_set.append(list(comp))
            unvisited = [n for n in self.graph if n not in visited and n not in comp]
            visited.update(comp)

        vis = cv2.cvtColor(self._img_skel, cv2.COLOR_GRAY2BGR)
        for branch in self.bridges:
            for y, x in branch:
                vis[y, x] = (0, 0, 255)
        for loop in self.loops:
            for y, x in loop:
                vis[y, x] = (0, 255, 0)
        for blob in core_set:
            for y, x in blob:
                vis[y, x] = (255, 0, 0)
        _save_step("4_segments", vis, self.out_dir)

    def _prune(self) -> None:
        pruned = self._img_skel.copy()
        for branch in self.bridges:
            for y, x in branch:
                pruned[y, x] = 0
        _save_step("5_pruned", pruned, self.out_dir)
        self._img_skel = pruned  # 覆寫為乾淨骨架

    def run_once(self, bin01: np.ndarray, do_prune: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        對「單次」輸入做骨架與拓撲分析；回傳 (pruned_skeleton_01, report)
        """
        self._from_binary01(bin01)
        self._skeletonize()
        self._build_graph()
        self._classify_nodes()
        self._separate_components()
        if do_prune:
            self._prune()

        report = dict(
            endpoints=len(self.endpoints),
            branch_points=len(self.branch_points),
            bridges=len(self.bridges),
            loops=len(self.loops),
            pruned=do_prune,
        )
        out01 = (self._img_skel > 0).astype(np.uint8)
        return out01, report

# ------------------------------------------------------------
# 新增：合成（大元件 + 指派的小元件）→ ROI 遮罩
# (相關函式已移除：compose_masks_for_large)
# ------------------------------------------------------------

# ------------------------------------------------------------
# 新增：對任意 ROI（0/1）做多回合骨架化/拓撲/修剪
# ------------------------------------------------------------
def process_roi_iter(roi01: np.ndarray, iterations: int, out_root: Optional[Path]) -> np.ndarray:
    """
    對輸入 ROI（0/1）做多回合「骨架化+拓撲分析+分支修剪」。回傳 0/1 骨架。
    """
    cur = roi01.copy().astype(np.uint8)
    iterations = max(1, int(iterations))
    for it in range(iterations):
        out_dir = None
        if out_root is not None:
            out_dir = out_root / f"iter_{it+1}"
        analyzer = TopologyAnalyzerArray(out_dir=out_dir)
        cur, _ = analyzer.run_once(cur, do_prune=True)
        if np.count_nonzero(cur) == 0:
            break
    return cur  # 0/1

# ------------------------------------------------------------
# 主要流程（已改為：僅處理 TopK 大元件）
# ------------------------------------------------------------
@timer
@show_memory("主要流程")
def run_pipeline(
    input_path: str,
    output_dir: str = DEFAULTS["OUTPUT_DIR"],
    top_n: int = DEFAULTS["TOP_N"],
    remove_largest: bool = DEFAULTS["REMOVE_LARGEST"],
    iterations: int = DEFAULTS["ITERATIONS"],
    save_steps: bool = DEFAULTS["SAVE_STEPS"],
    simplify_output: bool = False, # <-- [新參數] 新增參數以簡化輸出
) -> None:
    """
    無 CLI，直接呼叫此函式。
    - Step 1: 連通元件分析、分拆與儲存、排序、定義大元件 (TopK)
    - Step 2: (已移除) 小元件指派邏輯
    - Step 3: 對每個大元件的「原始遮罩 ROI」，做骨架化+拓撲分析+分支修剪（重複 iterations 次）
    - Step 4: 輸出每個大元件處理結果，以及全圖合成（含黑底與原底色版本）
    
    simplify_output (bool):
        若為 True，將只輸出「與原圖底色相同」的最終合成圖，
        並且檔名將會與 input_path 相同，儲存於 output_dir。
        所有其他中間檔案（components_raw, large_processed, merged/*）將不會被儲存。
    """
    output_root = Path(output_dir)
    
    # --- [修改] 根據 simplify_output 決定路徑和行為 ---
    if simplify_output:
        # 簡化模式：只需要 output_root，且強制關閉 save_steps
        ensure_dir(output_root)
        comp_dir = None
        large_dir = None
        merged_dir = output_root  # 最終輸出到根目錄
        save_steps = False # 強制關閉中間步驟
    else:
        # 原始模式：建立完整子目錄
        comp_dir = output_root / "components_raw"
        large_dir = output_root / "large_processed"
        merged_dir = output_root / "merged"
        for d in [comp_dir, large_dir, merged_dir]:
            ensure_dir(d)
    # --- [修改結束] ---

    # 讀檔 + 自動二值化（支援白/黑底）
    src = cv2.imread(input_path)
    if src is None:
        raise FileNotFoundError(f"無法讀取影像：{input_path}")
    bw01, bg_mode = auto_binarize(src)  # 前景=1，bg_mode={"white","black"}

    # Step 1: 連通元件、排序、定義大/小
    comps = analyze_components(bw01)
    large, small = select_large_small(comps, top_n, remove_largest) # 'small' 現在未被使用

    # --- [修改] 只有在非簡化模式下才儲存
    if not simplify_output:
        # 輸出拆分後之單一元件（未處理，便於檢視）
        for comp in comps:
            x, y, w, h = comp.bbox
            roi_src = src[y:y+h, x:x+w]
            roi_mask = comp.mask[y:y+h, x:x+w]
            canvas = np.full_like(roi_src, (255, 255, 255))
            canvas = np.where(roi_mask[:, :, None] == 1, roi_src, canvas)
            cv2.imwrite(str(comp_dir / f"component_{comp.label}_area{comp.area}.png"), canvas)

    # Step 2: (已移除) 小元件指派

    # Step 3:（關鍵改動）僅處理大元件，對其原始遮罩做骨架化/拓撲/修剪（迭代）
    large_skel_roi: Dict[int, np.ndarray] = {}
    for L in large:
        
        # --- [修改] 決定 L_out 和 L_out_steps
        L_out = None
        L_out_steps = None
        if not simplify_output:
            L_out = large_dir / f"L_{L.label}_area{L.area}"
            L_out_steps = L_out if save_steps else None
            ensure_dir(L_out)
        # --- [修改結束] ---

        # 3.1 取得大元件的裁切遮罩 (ROI)
        x, y, w, h = L.bbox
        original_roi_mask = L.mask[y:y+h, x:x+w].copy().astype(np.uint8)  # 0/1
        
        # --- [修改] 條件化儲存
        if not simplify_output:
            cv2.imwrite(str(L_out / "0_original_mask.png"), (original_roi_mask * 255).astype(np.uint8))

        # 3.2 對「原始遮罩」做多回合「骨架化+拓撲分析+分支修剪」
        # L_out_steps 在 simplify_output=True 時必為 None
        skel_roi = process_roi_iter(original_roi_mask, iterations, L_out_steps)
        large_skel_roi[L.label] = skel_roi
        
        # --- [修改] 條件化儲存
        if not simplify_output:
            cv2.imwrite(str(L_out / "pruned_skeleton.png"), (skel_roi * 255).astype(np.uint8))

    # Step 4: 全圖合成與輸出
    H, W = bw01.shape
    all_merged = np.zeros((H, W), dtype=np.uint8)
    for L in large:
        x, y, w, h = L.bbox
        roi = (large_skel_roi[L.label] * 255).astype(np.uint8)
        all_merged[y:y+h, x:x+w] = np.maximum(all_merged[y:y+h, x:x+w], roi)
        
        # --- [修改] 條件化儲存
        if not simplify_output:
            # 另輸出每個大元件的 ROI 成果（便於逐顆檢視）
            cv2.imwrite(str(merged_dir / f"merged_L{L.label}_area{L.area}.png"), roi)

    # --- [修改] 條件化儲存
    if not simplify_output:
        # ---- 全圖（黑底骨架） ----
        cv2.imwrite(str(merged_dir / f"merged_all_large.png"), all_merged)

    # ---- 另外輸出：與原圖底色相同圖（自動判斷白/黑底，筆畫色相反）----
    if bg_mode == "white":
        bg_val, fg_val = 255, 0  # 白底、黑線
    else:
        bg_val, fg_val = 0, 255  # 黑底、白線
    colored = np.full((H, W, 3), bg_val, dtype=np.uint8)
    idx = all_merged > 0
    colored[idx] = fg_val
    
    # --- [修改] 根據 simplify_output 決定最終儲存路徑 ---
    if simplify_output:
        # 簡化模式：使用輸入檔名，儲存到 output_dir 根目錄
        input_basename = Path(input_path).name
        final_output_path = output_root / input_basename
        cv2.imwrite(str(final_output_path), colored)
    else:
        # 原始模式：儲存到 merged 資料夾
        cv2.imwrite(str(merged_dir / f"merged_all_large_bgMatch.png"), colored)
    # --- [修改結束] ---

    # ---- 總結輸出 ----
    out_root_str = str(output_root)
    big_list = [{'label': L.label, 'area': L.area} for L in large]

    print("[Pipeline] Done.")
    
    # --- [修改] 根據模式顯示不同的總結訊息 ---
    if simplify_output:
        input_basename = Path(input_path).name
        final_output_path = output_root / input_basename
        print(f"  [簡化模式] 輸出檔案：{final_output_path}")
        print(f"  處理元件數：{len(big_list)}")
    else:
        print(f"  輸出資料夾：{out_root_str}")
        print(f"  大元件數 (TopK)：{len(big_list)}")
        for b in big_list:
            print(f"    - big label {b['label']:>3}  area={b['area']:>8}")
    # --- [修改結束] ---

# ------------------------------------------------------------
# 範例（供引用端參考）
# ------------------------------------------------------------
if __name__ == "__main__":
    # 注意：本模組不提供 CLI；以下段落僅作為「在 IDE 中直接執行」的測試入口。
    
    # --- [修改] 範例 1：完整輸出模式 ---
    print("--- 執行範例 1 (完整模式) ---")
    sample_input_1 = "data/engineering_images_100dpi_flat/train/刀庫底座__2L0T-LB50012-1070000_page_1.png"
    run_pipeline(
        input_path=sample_input_1,
        output_dir="results/test/0", # <-- 更改輸出目錄以示區別
        top_n=6,
        remove_largest=True,
        iterations=1,  # ⇠ 可視雜訊調整（例如 3、5、10…）
        save_steps=False,
        simplify_output=False, # <-- 設為 False (預設)
    )
    
    print("\n" + "="*50 + "\n")
    
    # --- [修改] 範例 2：簡化輸出模式 ---
    print("--- 執行範例 2 (簡化模式) ---")
    sample_input_2 = "data/engineering_images_100dpi_flat/train/刀庫底座__AW0BCH0C-50060030100_page_1.png"
    output_dir_2 = "results/test/simplified" # <-- 使用一個新的目錄來存放簡化後的
    
    run_pipeline(
        input_path=sample_input_2,
        output_dir=output_dir_2, # <-- 簡化模式的輸出目錄
        top_n=6,
        remove_largest=True,
        iterations=1,
        save_steps=False, # (在簡化模式下也會被強制為 False)
        simplify_output=True, # <-- [新參數] 觸發新行為
    )
    
    # 輔助提示，告知使用者檔案在哪
    input_basename_2 = Path(sample_input_2).name
    print(f"-> 簡化模式輸出應位於：{Path(output_dir_2) / input_basename_2}")