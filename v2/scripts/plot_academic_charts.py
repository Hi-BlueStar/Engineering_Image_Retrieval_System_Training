import argparse
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
    
    # 將空格轉換為底線的小寫檔案名稱
    file_prefix = metric_name.lower().replace(" ", "_")
    
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


def main():
    parser = argparse.ArgumentParser(description="生成學術期刊風格的檢索相似度/距離分析圖表")
    parser.add_argument("--input", type=str, required=True, help="評估腳本輸出的 CSV 檔案路徑 (例如: similarity_scores.csv)")
    parser.add_argument("--output_dir", type=str, default="results/figures", help="圖表輸出的資料夾路徑")
    
    args = parser.parse_args()
    csv_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not csv_path.exists():
        print(f"錯誤: 找不到指定的輸入檔案 '{csv_path}'")
        return
        
    print(f"載入資料: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if "is_tp" not in df.columns or "similarity_score" not in df.columns:
        print("錯誤: CSV 檔案必須包含 'is_tp' 與 'similarity_score' 欄位。")
        return
        
    tp_sim = df[df["is_tp"] == "True Positive"]["similarity_score"].values
    fp_sim = df[df["is_tp"] == "False Positive"]["similarity_score"].values
    
    setup_academic_style()
    
    # 1. 繪製 Cosine Similarity 分佈
    print("\n開始繪製 Cosine Similarity 分佈圖...")
    plot_distribution(
        tp_scores=tp_sim,
        fp_scores=fp_sim,
        metric_name="Cosine Similarity",
        output_dir=output_dir,
        bins=np.linspace(-1.0, 1.0, 100),
        x_lim=[-1.05, 1.05],
        xlabel_name="Cosine Similarity"
    )
    
    # 2. 轉換為 Normalized L2 Distance 並繪製
    # 對於 L2 Normalized 特徵向量: L2_dist = sqrt(2 - 2 * cosine_sim)
    print("\n開始計算並繪製 Normalized L2 Distance 分佈圖...")
    # np.clip 防止因浮點數精度問題導致根號內出現微小負數
    tp_l2 = np.sqrt(np.clip(2.0 - 2.0 * tp_sim, 0.0, None))
    fp_l2 = np.sqrt(np.clip(2.0 - 2.0 * fp_sim, 0.0, None))
    
    plot_distribution(
        tp_scores=tp_l2,
        fp_scores=fp_l2,
        metric_name="Normalized L2 Distance",
        output_dir=output_dir,
        bins=np.linspace(0.0, 2.0, 100),
        x_lim=[-0.05, 2.05],
        xlabel_name="L2 Distance (Normalized)"
    )

if __name__ == "__main__":
    main()
