#!/usr/bin/env python3
"""實驗計畫書驗證集 (V) 與訓練集 (T_small, T_large) 輔助建構與瀏覽工具。

此工具實現了實驗計畫書第三節「驗證集 (V) 建構」的四個步驟：
1. 種子抽樣：隨機抽取 50 張有標籤影像作為 Query Seeds。
2. 語義特徵檢索：利用 pre-trained/checkpoint 提取高維特徵並檢索相似影像建立候選池。
3. 專家篩選：透過 Streamlit UI 輔助研究員瀏覽並勾選 Ground Truth (GT)。
4. 背景雜訊混入與資料集劃分：混入 Distractors 生成 V，並劃分出 T_small 與 T_large。
"""

"""
uv run streamlit run v2/scripts/validation_set_builder.py
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

# 確保專案根目錄在 sys.path 中
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_filename_template(name: str) -> str:
    """將檔名轉換為結構模板，例如 '082B40-HOH-000-11500' -> 'DDDLDD-LLL-DDD-DDDDD'"""
    base = Path(name).name.split('.')[0]
    template = []
    for char in base:
        if char.isdigit():
            template.append('D')
        elif char.isalpha():
            template.append('L')
        else:
            template.append(char)
    return "".join(template)

# ----------------------------------------------------------------------
# 樣式與頁面設定已移至 main() 以防在 bare Python 模式下執行或導入時拋出警告
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 快取資料集定義
# ----------------------------------------------------------------------
class InferenceDataset(Dataset):
    def __init__(self, file_list: list[Path], img_size: int = 256):
        self.file_list = file_list
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert("L")
                return self.transform(img), str(img_path)
        except Exception:
            # 發生毀損圖檔時回傳全白圖
            return torch.ones(1, 256, 256), str(img_path)

# ----------------------------------------------------------------------
# 核心邏輯函數
# ----------------------------------------------------------------------
@st.cache_resource
def get_model(checkpoint_path: str = None, device: torch.device = torch.device("cpu")):
    """載入特徵提取模型。"""
    # 建立 ResNet18 特徵提取器
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # 取代全連接層
    model.fc = nn.Identity()

    if checkpoint_path and Path(checkpoint_path).is_file():
        try:
            st.toast(f"正在從 Checkpoint 載入特徵骨幹: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            state_dict = state.get("state_dict", state)
            # 清理 _orig_mod 前綴與 projector/predictor 權重，僅保留 backbone
            backbone_state = {}
            for k, v in state_dict.items():
                k_clean = k.replace("_orig_mod.backbone.", "").replace("backbone.", "")
                if not k.startswith("projector") and not k.startswith("predictor"):
                    backbone_state[k_clean] = v
            model.load_state_dict(backbone_state, strict=False)
        except Exception as e:
            st.error(f"載入 Checkpoint 失敗 ({e})，將使用隨機權重。")
    else:
        st.toast("未提供 checkpoint 或檔案不存在，使用隨機初始化 Backbone 進行特徵提取。")

    model.to(device)
    model.eval()
    return model

def extract_all_features(model, file_paths: list[Path], batch_size: int, device: torch.device):
    """提取給定檔案列表的特徵向量。"""
    dataset = InferenceDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    features = []
    paths = []

    progress_bar = st.progress(0.0, text="正在提取特徵...")
    total = len(dataloader)

    with torch.no_grad():
        for idx, (imgs, batch_paths) in enumerate(dataloader):
            imgs = imgs.to(device)
            feats = model(imgs)
            # L2 正規化以便計算餘弦相似度
            feats = torch.nn.functional.normalize(feats, dim=1)
            features.append(feats.cpu().numpy())
            paths.extend(batch_paths)
            progress_bar.progress((idx + 1) / total, text=f"特徵提取進度: {idx + 1}/{total} batches")

    progress_bar.empty()
    return np.concatenate(features, axis=0), paths

def build_symlinks(mapping: dict, target_dir: Path):
    """根據對照表建立符號連結以節省磁碟空間。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    for category, paths in mapping.items():
        cat_dir = target_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for p_str in paths:
            p = Path(p_str)
            link_path = cat_dir / p.name
            if link_path.is_symlink() or link_path.exists():
                try:
                    link_path.unlink()
                except Exception:
                    pass
            try:
                import os
                rel_path = os.path.relpath(p.resolve(), start=link_path.parent)
                link_path.symlink_to(rel_path)
            except Exception:
                # Windows 或不支援 symlink 環境下改用拷貝
                import shutil
                try:
                    shutil.copy(p, link_path)
                except Exception:
                    pass

# ----------------------------------------------------------------------
# Streamlit 介面與狀態管理
# ----------------------------------------------------------------------
def main():
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        print("This script is a Streamlit application and cannot be run directly with python.")
        print("Please run it using: streamlit run v2/scripts/validation_set_builder.py")
        sys.exit(1)

    # 樣式與頁面設定 (Rich Premium Aesthetics)
    st.set_page_config(
        page_title="SimSiam 驗證集與實驗劃分工具",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .main {
            background-color: #0F172A;
            color: #F8FAFC;
        }
        .stApp {
            background-color: #0F172A;
        }
        .sidebar .sidebar-content {
            background-color: #1E293B;
        }
        h1, h2, h3 {
            color: #38BDF8 !important;
            font-family: 'Outfit', 'Inter', sans-serif;
        }
        .stCheckbox > label {
            color: #E2E8F0 !important;
            font-size: 14px;
        }
        .card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            backdrop-filter: blur(10px);
        }
        .seed-title {
            color: #F43F5E;
            font-weight: bold;
            font-size: 18px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #10B981;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🔍 SimSiam 驗證集 (V) 與實驗劃分輔助系統")
    st.write("依據《實驗計畫書》第三章，使用自監督模型特徵建立候選池，並由研究人員進行篩選以建立客觀 Ground Truth。")

    # 1. 側邊欄設定區
    st.sidebar.header("📁 目錄與參數設定")
    da_dir = st.sidebar.text_input("Da (有標籤資料夾路徑)", "data/converted_labeled_images")
    db_dir = st.sidebar.text_input("Db (無標籤資料夾路徑)", "data/converted_images")
    output_dir_str = st.sidebar.text_input("實驗資料集輸出根目錄", "dataset_v2")
    checkpoint_path = st.sidebar.text_input("自定義 Checkpoint 路徑 (可留空)", "models/checkpoint_best_0312.pth")

    n_seeds = st.sidebar.slider("種子 (Seeds) 數量", 10, 100, 50)
    k_candidates = st.sidebar.slider("候選池 (Candidates) 數量 / Seed", 10, 100, 100)
    n_distractors = st.sidebar.number_input("混入干擾項 (Distractors) 數量", value=800)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 Session State
    if "seeds" not in st.session_state:
        st.session_state.seeds = []
    if "candidates" not in st.session_state:
        st.session_state.candidates = {}  # seed_path -> list of (candidate_path, score)
    if "gt_selections" not in st.session_state:
        st.session_state.gt_selections = {}  # seed_path -> list of chosen_candidate_paths
    if "reviewed_seeds" not in st.session_state:
        st.session_state.reviewed_seeds = set()
    if "all_paths_da" not in st.session_state:
        st.session_state.all_paths_da = []
    if "all_paths_db" not in st.session_state:
        st.session_state.all_paths_db = []

    # 頁面 Tabs
    tab1, tab2, tab3 = st.tabs(["⚡ 步驟 1 & 2: 種子抽樣與檢索", "🎨 步驟 3: 專家交互篩選 (GT)", "💾 步驟 4: 輸出與劃分"])

    # ------------------------------------------------------------------
    # Tab 1: 種子抽樣與特徵檢索
    # ------------------------------------------------------------------
    with tab1:
        st.header("Step 1 & 2: 種子抽樣與特徵檢索")
        st.write("自 $D_a$ 隨機均勻抽樣種子影像，並利用模型提取特徵，與 $D_a \\cup D_b$ 比對計算餘弦相似度，建立相似圖形候選池。")

        if st.button("🚀 啟動種子抽樣與候選池檢索"):
            path_da = Path(da_dir)
            path_db = Path(db_dir)

            if not path_da.is_dir() or not path_db.is_dir():
                st.error("請確認輸入的 Da 與 Db 資料夾路徑正確！")
                return

            # 掃描所有影像
            img_exts = {".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            all_da = sorted([p for p in path_da.rglob("*") if p.is_file() and p.suffix.lower() in img_exts])
            all_db = sorted([p for p in path_db.rglob("*") if p.is_file() and p.suffix.lower() in img_exts])

            st.session_state.all_paths_da = [str(p) for p in all_da]
            st.session_state.all_paths_db = [str(p) for p in all_db]

            st.info(f"掃描完成: Da 包含 {len(all_da)} 張標註影像, Db 包含 {len(all_db)} 張無標籤影像")

            # 1. 種子抽樣
            random.seed(42)  # 固定 Seed 確保重現性
            seeds = random.sample(all_da, min(n_seeds, len(all_da)))
            st.session_state.seeds = [str(p) for p in seeds]
            st.success(f"已隨機均勻抽出 {len(seeds)} 個 Query 種子影像。")

            # 2. 語義特徵檢索
            model = get_model(checkpoint_path, device)

            # 合併所有影像並提取特徵
            all_files = all_da + all_db
            st.info(f"開始提取總共 {len(all_files)} 張影像的特徵向量...")
            features, paths = extract_all_features(model, all_files, batch_size=128, device=device)

            # 建立路徑到特徵索引的對照
            path_to_idx = {p: i for i, p in enumerate(paths)}

            # 針對每個種子計算相似度並取得候選池
            candidates = {}
            for seed_path in st.session_state.seeds:
                s_idx = path_to_idx.get(seed_path)
                if s_idx is None:
                    continue
                seed_feat = features[s_idx] # [D,]

                # 計算與所有影像的餘弦相似度
                # features: [N, D]，與 seed_feat [D,] 做內積
                sims = np.dot(features, seed_feat) # [N,]

                # 排除種子本身
                sims[s_idx] = -1.0

                # 排序取得 top-K
                top_indices = np.argsort(sims)[::-1][:k_candidates]

                cand_list = []
                for idx_t in top_indices:
                    cand_list.append((paths[idx_t], float(sims[idx_t])))
                candidates[seed_path] = cand_list

            st.session_state.candidates = candidates
            st.session_state.gt_selections = {s: [] for s in st.session_state.seeds}
            st.session_state.reviewed_seeds = set()
            st.success("候選池檢索完成！請切換至第二分頁「專家交互篩選」進行人工標記。")

        # 顯示當前狀態
        if st.session_state.seeds:
            st.subheader(f"已抽樣之 Query Seeds ({len(st.session_state.seeds)} 張)")
            cols = st.columns(5)
            for i, s_path in enumerate(st.session_state.seeds[:15]):
                with cols[i % 5]:
                    st.image(s_path, caption=f"Seed #{i+1}\n{Path(s_path).name}", use_container_width=True)
            if len(st.session_state.seeds) > 15:
                st.write(f"... 還有 {len(st.session_state.seeds) - 15} 個種子未完全展開顯示。")

    # ------------------------------------------------------------------
    # Tab 2: 專家交互篩選 (GT 標定)
    # ------------------------------------------------------------------
    with tab2:
        st.header("Step 3: 專家篩選與 Ground Truth 確立")
        st.write("由人類研究員檢視高維相似度檢索候選池。點選符合「客觀幾何形狀相似性」的影像，確立為該 Query 的 Ground Truth (GT)。")

        if not st.session_state.seeds:
            st.warning("請先在第一個分頁完成「種子抽樣與檢索」！")
        else:
            # 建立下拉式選單選擇要標註的 Seed
            seed_options = [f"Seed #{i+1}: {Path(s).name} ({'已審查' if s in st.session_state.reviewed_seeds else '待審查'})" for i, s in enumerate(st.session_state.seeds)]
            selected_idx = st.selectbox("選擇要瀏覽的 Query 種子", range(len(seed_options)), format_func=lambda x: seed_options[x])

            selected_seed = st.session_state.seeds[selected_idx]

            # 主版面分兩欄：左欄顯示 Seed，右欄顯示候選池
            col_left, col_right = st.columns([1, 3])

            seed_template = get_filename_template(selected_seed)
            cand_data = st.session_state.candidates.get(selected_seed, [])

            with col_left:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"<span class='seed-title'>當前 Query 種子 (Seed #{selected_idx+1})</span>", unsafe_allow_html=True)
                st.image(selected_seed, use_container_width=True)
                st.write(f"**路徑**: `{selected_seed}`")
                st.markdown(f"**結構模板**: `{seed_template}`")
                st.markdown("</div>", unsafe_allow_html=True)

                # 輔助篩選機制
                st.markdown("### 🔍 檔名結構輔助篩選")
                score_threshold = st.slider("相似度閾值過濾", 0.0, 1.0, 0.0, 0.05)

                filter_option = st.radio(
                    "篩選模式",
                    ["顯示全部", "僅顯示與 Seed 相同結構模板", "自訂結構模板篩選", "按橫線段數篩選"]
                )

                templates_in_candidates = sorted(list({get_filename_template(cand[0]) for cand in cand_data}))
                parts_count_in_candidates = sorted(list({len(Path(cand[0]).name.split('.')[0].split('-')) for cand in cand_data}))

                selected_template = None
                selected_parts = None

                if filter_option == "自訂結構模板篩選":
                    selected_template = st.selectbox("選擇結構模板", templates_in_candidates)
                elif filter_option == "按橫線段數篩選":
                    selected_parts = st.selectbox("選擇段數 (以 '-' 分隔)", parts_count_in_candidates)

                # 預先篩選出符合條件的影像路徑，以供批量操作與顯示
                filtered_cands = []
                for cand_path, score in cand_data:
                    if score < score_threshold:
                        continue
                    cand_template = get_filename_template(cand_path)
                    cand_parts = len(Path(cand_path).name.split('.')[0].split('-'))

                    if filter_option == "僅顯示與 Seed 相同結構模板" and cand_template != seed_template or filter_option == "自訂結構模板篩選" and cand_template != selected_template or filter_option == "按橫線段數篩選" and cand_parts != selected_parts:
                        continue
                    filtered_cands.append(cand_path)

                st.markdown("⚡ **批量操作**")
                col_btn1, col_btn2 = st.columns(2)
                if col_btn1.button("✅ 全選顯示", help="將目前畫面上顯示的所有候選影像加入 GT"):
                    for path in filtered_cands:
                        if path not in st.session_state.gt_selections[selected_seed]:
                            st.session_state.gt_selections[selected_seed].append(path)
                    st.toast("已全選目前顯示的候選影像！")
                    st.rerun()

                if col_btn2.button("❌ 全消顯示", help="將目前畫面上顯示的所有候選影像移出 GT"):
                    for path in filtered_cands:
                        if path in st.session_state.gt_selections[selected_seed]:
                            st.session_state.gt_selections[selected_seed].remove(path)
                    st.toast("已取消選取目前顯示的候選影像！")
                    st.rerun()

                st.write("---")
                # 進度顯示
                reviewed_count = len(st.session_state.reviewed_seeds)
                total_seeds = len(st.session_state.seeds)
                st.write("### 審查進度")
                st.write(f"已審查: **{reviewed_count} / {total_seeds}**")
                st.progress(reviewed_count / total_seeds)

                if st.button("💾 保存當前 Seed 的勾選", type="primary"):
                    st.session_state.reviewed_seeds.add(selected_seed)
                    st.toast("已成功保存該種子的 GT 勾選狀態！")
                    st.rerun()

            with col_right:
                st.subheader(f"語義特徵相似候選池 (Top {k_candidates}) — 勾選符合幾何形狀相似者作為 Ground Truth")

                if not cand_data:
                    st.info("該種子無候選影像數據。")
                else:
                    # 篩選並組裝需要顯示的 candidate data
                    filtered_cand_data = []
                    for cand_path, score in cand_data:
                        if cand_path in filtered_cands:
                            filtered_cand_data.append((cand_path, score))

                    if not filtered_cand_data:
                        st.info("沒有符合當前篩選條件的候選影像。")
                    else:
                        st.write(f"目前顯示: **{len(filtered_cand_data)} / {len(cand_data)}** 個候選影像")

                        import hashlib
                        # 每列顯示 4 張影像
                        cols_c = st.columns(4)
                        for c_idx, (cand_path, score) in enumerate(filtered_cand_data):
                            with cols_c[c_idx % 4]:
                                # 若此候選影像的模板與 Seed 相同，給予綠色邊框特別強調
                                cand_template = get_filename_template(cand_path)
                                is_same_template = cand_template == seed_template
                                border_style = "border: 2px solid #10B981;" if is_same_template else "border: 1px solid rgba(255, 255, 255, 0.1);"

                                st.markdown(f"<div class='card' style='{border_style}'>", unsafe_allow_html=True)
                                st.image(cand_path, use_container_width=True)
                                st.write(f"相似度: `{score:.4f}`")

                                if is_same_template:
                                    st.markdown(f"<span style='color:#10B981; font-weight:bold; font-size:12px;'>✨ 模板符合: `{cand_template}`</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='color:#94A3B8; font-size:12px;'>模板: `{cand_template}`</span>", unsafe_allow_html=True)

                                st.write(f"檔名: `{Path(cand_path).name}`")

                                # 檢查先前是否已勾選
                                was_selected = cand_path in st.session_state.gt_selections[selected_seed]

                                # 使用 stable hash 的 key 避免 Streamlit widget state 混亂
                                path_hash = hashlib.md5(cand_path.encode()).hexdigest()[:8]
                                unique_key = f"check_{selected_idx}_{path_hash}"

                                is_checked = st.checkbox(
                                    "標記為 GT",
                                    value=was_selected,
                                    key=unique_key
                                )

                                # 動態更新勾選狀態到 session_state
                                if is_checked and cand_path not in st.session_state.gt_selections[selected_seed]:
                                    st.session_state.gt_selections[selected_seed].append(cand_path)
                                elif not is_checked and cand_path in st.session_state.gt_selections[selected_seed]:
                                    st.session_state.gt_selections[selected_seed].remove(cand_path)

                                st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Tab 3: 背景雜訊混入與劃分輸出
    # ------------------------------------------------------------------
    with tab3:
        st.header("Step 4: 背景雜訊混入與資料集劃分")
        st.write("在此階段，我們將選定的 Queries 與其對應的 Ground Truth (GT) 合併，並隨機抽取干擾項 (Distractors) 混合構成驗證集 $V$。同時將剩餘資料劃分為小資料集 $T_{small}$ 與大資料集 $T_{large}$。")

        if not st.session_state.seeds:
            st.warning("請先在第一個分頁完成「種子抽樣與檢索」！")
        else:
            # 統計目前的 GT 數量
            total_gts = sum(len(v) for v in st.session_state.gt_selections.values())
            st.write(f"- 共有 **{len(st.session_state.seeds)}** 張 Query Seeds。")
            st.write(f"- 研究員已勾選確立的 Ground Truth 共 **{total_gts}** 張。")

            # 設定輸出目錄
            out_root = Path(output_dir_str)
            st.write(f"- 預設輸出路徑: `{out_root.resolve()}`")

            if st.button("🚀 執行背景雜訊混入與生成 splits"):
                # 收集所有已選入 V 的影像
                all_v_images = set(st.session_state.seeds)
                for gt_list in st.session_state.gt_selections.values():
                    all_v_images.update(gt_list)

                # 計算剩餘未標記的影像，作為干擾項 (Distractors) 候選
                all_da = set(st.session_state.all_paths_da)
                all_db = set(st.session_state.all_paths_db)
                all_universe = all_da.union(all_db)

                remaining_candidates = list(all_universe.difference(all_v_images))

                # 隨機抽取指定數量的 distractors
                random.seed(42)
                chosen_distractors = random.sample(
                    remaining_candidates,
                    min(n_distractors, len(remaining_candidates))
                )

                # 驗證集 V 構成：Queries + GTs + Distractors
                all_v_images.update(chosen_distractors)

                # 輸出劃分定義：
                # T_small = Da \ V
                t_small_images = list(all_da.difference(all_v_images))

                # T_large = (Da U Db) \ V
                t_large_images = list(all_universe.difference(all_v_images))

                st.success("驗證集與訓練集劃分計算完成！")

                # 顯示統計數據卡片
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.write("**驗證集 V 大小**")
                    st.markdown(f"<span class='metric-value'>{len(all_v_images)}</span> 張", unsafe_allow_html=True)
                    st.write(f"(含 {len(st.session_state.seeds)} Queries, {total_gts} GTs, {len(chosen_distractors)} Distractors)")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_m2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.write("**T_small (小訓練集) 大小**")
                    st.markdown(f"<span class='metric-value'>{len(t_small_images)}</span> 張", unsafe_allow_html=True)
                    st.write("($D_a \\setminus V$)")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_m3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.write("**T_large (大訓練集) 大小**")
                    st.markdown(f"<span class='metric-value'>{len(t_large_images)}</span> 張", unsafe_allow_html=True)
                    st.write("($(D_a \\cup D_b) \\setminus V$)")
                    st.markdown("</div>", unsafe_allow_html=True)

                # 建立並儲存 metadata.json
                meta = {
                    "seeds": st.session_state.seeds,
                    "gt_selections": st.session_state.gt_selections,
                    "distractors": chosen_distractors,
                    "V": list(all_v_images),
                    "T_small": t_small_images,
                    "T_large": t_large_images
                }

                out_root.mkdir(parents=True, exist_ok=True)
                meta_path = out_root / "validation_split.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=4, ensure_ascii=False)
                st.info(f"對照表已寫入: `{meta_path}`")

                # 建立實體符號連結結構 (Symlinks) 供訓練腳本直接讀取
                st.info("正在建立對應實驗資料集目錄（以符號連結避免冗餘複製）...")

                # 建立 T_small 的訓練/測試切分 (按 8:2 劃分)
                random.seed(42)
                random.shuffle(t_small_images)
                split_idx_s = int(len(t_small_images) * 0.8)
                t_small_train = t_small_images[:split_idx_s]
                t_small_test = t_small_images[split_idx_s:]

                build_symlinks({"Component_Dataset/train": t_small_train, "Component_Dataset/test": t_small_test}, out_root / "T_small" / "Run_01_Seed_42")

                # 建立 T_large 的訓練/測試切分 (按 8:2 劃分)
                random.shuffle(t_large_images)
                split_idx_l = int(len(t_large_images) * 0.8)
                t_large_train = t_large_images[:split_idx_l]
                t_large_test = t_large_images[split_idx_l:]

                build_symlinks({"Component_Dataset/train": t_large_train, "Component_Dataset/test": t_large_test}, out_root / "T_large" / "Run_01_Seed_42")

                # 建立 驗證集 V 結構（包含 Query 與 Gallery）
                # 每個 Query 擁有各自的子目錄以作為類別，避免分類資訊消失與指標平凡化
                v_mapping = {}
                for i, seed in enumerate(st.session_state.seeds):
                    group_name = f"group_{i:03d}"
                    group_images = [seed] + st.session_state.gt_selections.get(seed, [])
                    # 確保不重複
                    group_images = list(dict.fromkeys(group_images))
                    v_mapping[group_name] = group_images

                # 每個 Distractor 放入獨立資料夾以避免互相匹配
                for j, dist in enumerate(chosen_distractors):
                    v_mapping[f"distractor_{j:04d}"] = [dist]

                build_symlinks(v_mapping, out_root / "V")

                st.success("實驗資料集目錄結構與符號連結已成功建立！")
                st.balloons()

if __name__ == "__main__":
    main()
