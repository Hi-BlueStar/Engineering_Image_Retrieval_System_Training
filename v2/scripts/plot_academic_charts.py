import argparse
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def setup_academic_style():
    """配置符合學術期刊排版的 Matplotlib 參數。
    
    特別為後續 XeLaTeX 排版最佳化，確保高對比度與清晰度。
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # 數學字體風格近似 Times
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 14,
        # 邊框與網格設定 (清晰高對比)
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "grid.color": "gray",
        # 刻度設定
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        # PDF 匯出設定 (相容 XeLaTeX，不使用系統依賴字體轉曲線)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # 儲存圖片設定
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05
    })

def plot_distribution(tp_scores, fp_scores, metric_name, output_dir, bins, x_lim, xlabel_name):
    """通用的學術分佈圖表繪製函式 (Histogram + Boxplot)。"""
    # 計算統計量
    tp_mean, fp_mean = np.mean(tp_scores), np.mean(fp_scores)
    margin = abs(tp_mean - fp_mean)
    
    # 建立畫布與子圖 (比例分配: Histogram 占 80%, Boxplot 占 20%)
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
    
    ax_hist = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1], sharex=ax_hist)
    
    # -------------------------------------------------------------
    # 1. 繪製 Step Histogram (避免 KDE 平滑造成的假象，呈現真實離散分佈)
    # -------------------------------------------------------------
    # 繪製 FP (Inter-class) - 紅色系
    ax_hist.hist(fp_scores, bins=bins, density=True, histtype='stepfilled', 
                 alpha=0.2, color='#E63946', label='Inter-class (FP)')
    ax_hist.hist(fp_scores, bins=bins, density=True, histtype='step', 
                 linewidth=1.5, color='#E63946')
    
    # 繪製 TP (Intra-class) - 藍色系
    ax_hist.hist(tp_scores, bins=bins, density=True, histtype='stepfilled', 
                 alpha=0.2, color='#1D3557', label='Intra-class (TP)')
    ax_hist.hist(tp_scores, bins=bins, density=True, histtype='step', 
                 linewidth=1.5, color='#1D3557')

    # 標示 Mean Lines
    ax_hist.axvline(fp_mean, color='#E63946', linestyle='--', linewidth=2, zorder=5)
    ax_hist.axvline(tp_mean, color='#1D3557', linestyle='--', linewidth=2, zorder=5)
    
    # 標示 Margin
    y_max = ax_hist.get_ylim()[1]
    y_arrow = y_max * 0.85
    ax_hist.annotate(
        f"Margin = {margin:.3f}", 
        xy=((tp_mean + fp_mean) / 2, y_arrow), 
        xytext=(0, 10), textcoords='offset points',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9)
    )
    # 繪製雙向箭頭
    ax_hist.annotate(
        '', xy=(fp_mean, y_arrow), xytext=(tp_mean, y_arrow),
        arrowprops=dict(arrowstyle='<->', color='black', linewidth=1.5)
    )

    ax_hist.set_ylabel("Probability Density")
    ax_hist.set_title(f"Discriminative Analysis: {metric_name} Distribution")
    ax_hist.legend(loc="upper left", framealpha=0.9, edgecolor="gray")
    ax_hist.grid(True)
    ax_hist.xaxis.set_minor_locator(AutoMinorLocator())
    ax_hist.yaxis.set_minor_locator(AutoMinorLocator())
    
    # 隱藏 histogram 的 x 軸標籤 (因為與底部的 boxplot 共用)
    plt.setp(ax_hist.get_xticklabels(), visible=False)

    # -------------------------------------------------------------
    # 2. 繪製 Boxplot (輔助觀察極端值與四分位數)
    # -------------------------------------------------------------
    box_data = [fp_scores, tp_scores]
    box_colors = ['#E63946', '#1D3557']
    
    bplot = ax_box.boxplot(box_data, vert=False, patch_artist=True, 
                           widths=0.6, showfliers=True, 
                           flierprops=dict(marker='o', markersize=3, alpha=0.3, markeredgecolor='none'))
    
    # 上色
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for median in bplot['medians']:
        median.set(color='black', linewidth=2)
    for whisker in bplot['whiskers']:
        whisker.set(color='black', linewidth=1.5, linestyle='-')
    for cap in bplot['caps']:
        cap.set(color='black', linewidth=1.5)

    ax_box.set_yticks([1, 2])
    ax_box.set_yticklabels(["Inter", "Intra"])
    ax_box.set_xlabel(xlabel_name)
    ax_box.set_xlim(x_lim)
    ax_box.grid(True, axis='x')
    ax_box.xaxis.set_minor_locator(AutoMinorLocator())

    # -------------------------------------------------------------
    # 儲存與匯出
    # -------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 將空格與斜線轉換為底線的小寫檔案名稱，避免路徑錯誤
    file_prefix = metric_name.lower().replace(" ", "_").replace("/", "_")
    
    # 匯出 PDF 與 EPS 供 XeLaTeX 使用，PNG 供快速預覽
    out_pdf = output_dir / f"{file_prefix}_distribution.pdf"
    out_eps = output_dir / f"{file_prefix}_distribution.eps"
    out_png = output_dir / f"{file_prefix}_distribution.png"
    
    fig.savefig(out_pdf)
    fig.savefig(out_eps)
    fig.savefig(out_png, dpi=300)
    
    print(f"✅ {metric_name} 圖表已成功匯出至:")
    print(f"  - {out_pdf}")
    print(f"  - {out_eps}")
    print(f"  - {out_png}")
    plt.close(fig)


def plot_training_curves(log_csv_path: Path, output_dir: Path):
    """繪製訓練進度曲線 (Loss 與 Feature Standard Deviation)。"""
    print(f"載入訓練日誌: {log_csv_path}")
    df = pd.read_csv(log_csv_path)
    
    if "epoch" not in df.columns:
        print("錯誤: 訓練日誌缺少 'epoch' 欄位。")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    # 1. Loss 曲線
    if "train_loss" in df.columns:
        ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color='#1D3557', linewidth=2)
    if "val_loss" in df.columns:
        ax1.plot(df["epoch"], df["val_loss"], label="Val Loss", color='#E63946', linewidth=2, linestyle='--')
    
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Progression: Loss Analysis")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Feature Standard Deviation 曲線 (SimSiam 關鍵指標，避免 Collapse)
    if "train_z_std" in df.columns:
        ax2.plot(df["epoch"], df["train_z_std"], label="Train Std", color='#1D3557', linewidth=2)
    if "val_z_std" in df.columns:
        ax2.plot(df["epoch"], df["val_z_std"], label="Val Std", color='#E63946', linewidth=2, linestyle='--')
    
    ax2.set_ylabel("Std Dev of $z$")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Feature Stability: Standard Deviation of Projector Output")
    ax2.legend()
    ax2.grid(True)
    
    # 設定 X 軸刻度
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    
    plt.tight_layout()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "training_curves.pdf")
    fig.savefig(output_dir / "training_curves.eps")
    fig.savefig(output_dir / "training_curves.png", dpi=300)
    print(f"✅ 訓練曲線圖已匯出至 {output_dir}")
    plt.close(fig)


def plot_metrics_summary(metrics: dict, output_dir: Path, title_suffix: str = ""):
    """繪製檢索指標摘要柱狀圖 (Precision@K)。"""
    # 提取 Precision@K 指標
    p_keys = [k for k in metrics.keys() if "precision" in k]
    
    # 排序 p_keys 以確保 P@1, P@5, P@10... 順序正確
    def get_k(key):
        match = re.search(r"top(\d+)", key)
        return int(match.group(1)) if match else 0
    
    p_keys.sort(key=get_k)
    p_values = [metrics[k] for k in p_keys]
    
    # 簡化標籤，例如 top1_precision -> P@1
    labels = [k.replace("_precision", "").replace("top", "P@") for k in p_keys]
    
    if not labels:
        print(f"未在指標字典中找到 Precision 指標{title_suffix}。")
        return

    fig, ax = plt.subplots(figsize=(10, 6))  # 稍微加寬以容納更多 K 值
    bars = ax.bar(labels, p_values, color='#457B9D', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # 在柱狀圖上方顯示數值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylim([0, max(p_values) * 1.2 if p_values else 1.1])
    ax.set_ylabel("Precision Score")
    ax.set_title(f"Retrieval Performance: Precision@K Summary{title_suffix}")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)  # 標籤旋轉避免重疊
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "metrics_summary.pdf")
    fig.savefig(output_dir / "metrics_summary.eps")
    fig.savefig(output_dir / "metrics_summary.png", dpi=300)
    print(f"✅ 指標摘要圖{title_suffix}已匯出至 {output_dir}")
    plt.close(fig)


def plot_similarity_analysis(input_path: Path, output_dir: Path, title_suffix: str = ""):
    """載入 CSV 並繪製相似度與距離分佈圖。"""
    if not input_path.exists():
        print(f"警告: 找不到輸入檔案 {input_path}")
        return

    print(f"\n載入資料: {input_path}")
    try:
        df = pd.read_csv(input_path)
        # 檢查必要欄位
        required_cols = ["is_tp", "similarity_score"]
        if not all(col in df.columns for col in required_cols):
            print(f"錯誤: CSV 檔案缺少必要欄位 {required_cols}")
            return

        tp_sim = df[df["is_tp"] == "True Positive"]["similarity_score"].values
        fp_sim = df[df["is_tp"] == "False Positive"]["similarity_score"].values
        
        if len(tp_sim) == 0 or len(fp_sim) == 0:
            # 嘗試檢查原始布林值或數值
            tp_sim = df[df["is_tp"].isin([True, 1, "1", "true", "True"])]["similarity_score"].values
            fp_sim = df[df["is_tp"].isin([False, 0, "0", "false", "False"])]["similarity_score"].values

        if len(tp_sim) > 0 and len(fp_sim) > 0:
            print(f"開始繪製 Cosine Similarity 分佈圖{title_suffix} (TP: {len(tp_sim)}, FP: {len(fp_sim)})...")
            plot_distribution(tp_sim, fp_sim, f"Cosine Similarity{title_suffix}", output_dir, 
                                np.linspace(-1.0, 1.0, 100), [-1.05, 1.05], "Cosine Similarity")
            
            print(f"開始計算並繪製 Normalized L2 Distance 分佈圖{title_suffix}...")
            tp_l2 = np.sqrt(np.clip(2.0 - 2.0 * tp_sim, 0.0, None))
            fp_l2 = np.sqrt(np.clip(2.0 - 2.0 * fp_sim, 0.0, None))
            plot_distribution(tp_l2, fp_l2, f"Normalized L2 Distance{title_suffix}", output_dir,
                                np.linspace(0.0, 2.0, 100), [-0.05, 2.05], "L2 Distance (Normalized)")
        else:
            print(f"警告: 無法從 CSV 中提取有效的 TP/FP 資料{title_suffix}。請檢查 'is_tp' 欄位內容。")
    except Exception as e:
        print(f"載入 CSV 失敗 ({input_path.name}): {e}")


def main():
    parser = argparse.ArgumentParser(description="生成學術期刊風格的檢索相似度/距離分析圖表")
    parser.add_argument("--input", type=str, help="相似度 CSV 檔案路徑 (similarity_scores.csv) 或評估結果目錄")
    parser.add_argument("--training_log", type=str, help="訓練日誌 CSV 檔案路徑 (training_log.csv)")
    parser.add_argument("--metrics", type=str, help="檢索指標 JSON 檔案路徑 (eval_results.json)")
    parser.add_argument("--output_dir", type=str, help="圖表輸出的資料夾路徑 (預設為 results/figures_exp_<timestamp>)")
    
    args = parser.parse_args()
    
    input_arg = Path(args.input) if args.input else None
    training_log = Path(args.training_log) if args.training_log else None
    metrics_path = Path(args.metrics) if args.metrics else None
    
    csv_paths = [] # 儲存所有待處理的 CSV
    
    # --- 1. 自動偵測與補完路徑 ---
    if input_arg:
        if input_arg.is_dir():
            print(f"輸入為目錄: {input_arg}，嘗試自動偵測檔案...")
            # 找 preprocessed 與 raw
            csv_paths.extend(list(input_arg.glob("*_preprocessed.csv")))
            csv_paths.extend(list(input_arg.glob("*_raw.csv")))
            if not csv_paths:
                csv_paths.extend(list(input_arg.glob("similarity_scores.csv")))
            
            json_candidates = list(input_arg.glob("eval_results.json")) + list(input_arg.glob("retrieval_metrics.json"))
            if json_candidates and not metrics_path:
                metrics_path = json_candidates[0]
                print(f"  -> 偵測到指標檔案: {metrics_path.name}")
        
        elif input_arg.suffix == ".json":
            if not metrics_path:
                print(f"偵測到輸入為 JSON 檔案，將其視為 --metrics: {input_arg}")
                metrics_path = input_arg
            # 嘗試在同目錄下尋找 CSV
            csv_paths.extend(list(input_arg.parent.glob("*_preprocessed.csv")))
            csv_paths.extend(list(input_arg.parent.glob("*_raw.csv")))
        
        elif input_arg.suffix == ".csv":
            csv_paths.append(input_arg)

    # --- 2. 自動推導時間戳記與輸出路徑 ---
    timestamp = None
    all_paths_to_check = csv_paths + ([training_log] if training_log else []) + ([metrics_path] if metrics_path else [])
    for p in all_paths_to_check:
        if p:
            match = re.search(r"exp_(\d{8}_\d{6})", str(p))
            if match:
                timestamp = match.group(1)
                break
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif timestamp:
        output_dir = Path("results") / f"figures_exp_{timestamp}"
    else:
        output_dir = Path("results/figures")
    
    print(f"輸出目錄設定為: {output_dir}")
    
    # --- 3. 嘗試自動尋找訓練日誌 (如果在 evaluate 目錄下) ---
    if not training_log and timestamp:
        outputs_root = Path("outputs")
        sim_exp_dir = outputs_root / f"simsiam_exp_{timestamp}"
        if sim_exp_dir.exists():
            log_candidates = list(sim_exp_dir.rglob("training_log.csv"))
            if log_candidates:
                training_log = log_candidates[0]
                print(f"自動偵測到訓練日誌: {training_log}")

    setup_academic_style()
    
    # 4. 相似度/距離分佈圖
    for csv_path in csv_paths:
        if "preprocessed" in csv_path.name:
            plot_similarity_analysis(csv_path, output_dir / "preprocessed", " (Preprocessed)")
        elif "raw" in csv_path.name:
            plot_similarity_analysis(csv_path, output_dir / "raw", " (Raw/No-Preprocess)")
        else:
            plot_similarity_analysis(csv_path, output_dir)

    # 5. 訓練曲線圖
    if training_log and training_log.exists():
        plot_training_curves(training_log, output_dir)

    # 6. 指標摘要圖
    if metrics_path and metrics_path.exists():
        print(f"\n載入指標檔案: {metrics_path}")
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            # 如果 JSON 包含多種模式，則分別繪製
            found_mode = False
            if "preprocessed" in data and "metrics" in data["preprocessed"]:
                plot_metrics_summary(data["preprocessed"]["metrics"], output_dir / "preprocessed", " (Preprocessed)")
                found_mode = True
            
            if "raw" in data and "metrics" in data["raw"]:
                plot_metrics_summary(data["raw"]["metrics"], output_dir / "raw", " (Raw/No-Preprocess)")
                found_mode = True
                
            if not found_mode:
                # 回退到舊邏輯：提取 metrics 字典
                metrics = data.get("metrics", data)
                # 某些格式可能直接是字典
                if isinstance(metrics, dict):
                    plot_metrics_summary(metrics, output_dir)
        except Exception as e:
            print(f"處理指標檔案失敗: {e}")

if __name__ == "__main__":
    main()
