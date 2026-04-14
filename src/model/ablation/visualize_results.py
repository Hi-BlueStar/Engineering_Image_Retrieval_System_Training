"""
TTA 實驗結果視覺化模組
------------------------------------------------------------------------------
本腳本讀取 `tta_gc_analysis.py` 產出的 CSV 結果，繪製以下圖表：
1. Global Drift Histogram: TTA 變體與原始圖的相似度分佈。
2. Centroid Gain Analysis: GC vs Single TTA 的效益比較。
3. Ablation Study Boxplot: 各 TTA 策略的相似度分佈比較。
4. Cohesion Scatter/Boxplot: 各 TTA 策略與 GC 的聚合程度。

Usage:
    python src/model/ablation/visualize_results.py --input_dir outputs/ablation_results
"""

import argparse
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# 設定中文字型 (嘗試常見系統字型，若無則回退英文)
def set_chinese_font():
    # 常見 Linux/Windows 中文字型列表
    fonts = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    for font in fonts:
        try:
            # 簡單測試是否可用 (matplotlib 查找)
            from matplotlib.font_manager import FontProperties
            FontProperties(fname=font) # 僅示意，實際通常設定 rcParams
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
            return
        except:
            continue
    print("Warning: Chinese font not found. Using default font.")

def main():
    parser = argparse.ArgumentParser(description="Visualize TTA Results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing CSV results")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 設定風格
    sns.set_theme(style="whitegrid")
    set_chinese_font()
    
    # -------------------------------------------------------------------------
    # 1. Similarity to Original(Global Drift Analysis) (Exp 1)
    # -------------------------------------------------------------------------
    # csv_exp1 = input_dir / "exp1_global_drift.csv"
    # if csv_exp1.exists():
    #     df1 = pd.read_csv(csv_exp1)
    #     plt.figure(figsize=(10, 6))
    #     sns.histplot(df1["similarity"], bins=30, kde=True, color="skyblue")
    #     plt.title("Similarity to Original")
    #     plt.xlabel("Cosine Similarity")
    #     plt.ylabel("Frequency")
    #     plt.axvline(x=0.85, color='r', linestyle='--', label='Threshold (0.85)')
    #     plt.legend()
    #     plt.savefig(output_dir / "exp1_global_drift.png")
    #     plt.close()
    #     print(f"Generated Exp 1 plot: {output_dir / 'exp1_global_drift.png'}")
    # else:
    #     print(f"Skipping Exp 1: {csv_exp1} not found.")
    csv_exp1 = input_dir / "exp1_global_drift.csv"
    if csv_exp1.exists():
        df1 = pd.read_csv(csv_exp1)
        data = df1["similarity"]
        
        # Calculate Statistics
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        min_val = data.min()
        max_val = data.max()
        
        stats_text = (f"Median: {median:.4f} | IQR: {iqr:.4f}\n"
                      f"Range: [{min_val:.4f}, {max_val:.4f}]")

        # --- Plot 1: Boxplot ---
        plt.figure(figsize=(10, 4))
        # Boxplot shows Median, IQR (box), and Outliers (points beyond whiskers) by default
        sns.boxplot(x=data, color="skyblue", flierprops={"marker": "x", "markerfacecolor": "red"})
        plt.title(f"Embedding Drift Analysis (Boxplot)\n{stats_text}")
        plt.xlabel("Cosine Similarity(TTA, Original)")
        plt.tight_layout()
        plt.xlim(0.0, 1.05)
        plt.savefig(output_dir / "exp1_global_drift_boxplot.png")
        plt.close()
        print(f"Generated Exp 1 Boxplot: {output_dir / 'exp1_global_drift_boxplot.png'}")

        # --- Plot 2: Violin Plot ---
        plt.figure(figsize=(10, 5))
        # Violin plot with 'box' inner to show quartiles/median similar to boxplot but with density
        sns.violinplot(x=data, color="lightgreen", inner="box")
        plt.title(f"Embedding Drift Analysis (Violin Plot)\n{stats_text}")
        plt.xlabel("Cosine Similarity(TTA, Original)")
        plt.tight_layout()
        plt.xlim(0.0, 1.05)
        plt.savefig(output_dir / "exp1_global_drift_violin.png")
        plt.close()
        print(f"Generated Exp 1 Violin Plot: {output_dir / 'exp1_global_drift_violin.png'}")
    else:
        print(f"Skipping Exp 1: {csv_exp1} not found.")

    # -------------------------------------------------------------------------
    # 2. Centroid Effectiveness (Exp 2)
    # -------------------------------------------------------------------------
    csv_exp2 = input_dir / "exp2_centroid.csv"
    if csv_exp2.exists():
        df2 = pd.read_csv(csv_exp2)
        # Calculate summary stats
        avg_gc = df2["sim_gc"].mean()
        avg_single = df2["avg_sim_single"].mean()
        
        plt.figure(figsize=(8, 6))
        # Melt for seaborn barplot
        df2_melt = pd.melt(df2[["sim_gc", "avg_sim_single"]], var_name="Method", value_name="Similarity")
        sns.boxplot(x="Method", y="Similarity", data=df2_melt, hue="Method", palette="Set2", legend=False)
        
        # Add labels
        plt.title("Centroid (GC) vs. Single TTA Effectiveness")
        plt.xlabel("Query Method")
        plt.ylabel("Similarity to Anchor (Original)")
        plt.xticks([0, 1], ["Geometry Center (GC)", "Avg Single TTA"])
        plt.ylim(0.7, 1.05)
        
        # Annotate means
        plt.text(0, avg_gc, f"{avg_gc:.4f}", ha='center', va='bottom', fontweight='bold')
        plt.text(1, avg_single, f"{avg_single:.4f}", ha='center', va='bottom', fontweight='bold')
        
        plt.savefig(output_dir / "exp2_centroid_effectiveness.png")
        plt.close()
        print(f"Generated Exp 2 plot: {output_dir / 'exp2_centroid_effectiveness.png'}")
    else:
        print(f"Skipping Exp 2: {csv_exp2} not found.")

    # 2.5.
    csv_exp2 = input_dir / "exp2_centroid.csv"
    if csv_exp2.exists():
        df2_5 = pd.read_csv(csv_exp2)
        data = df2_5["sim_gc"]

        # Calculate Statistics
        median = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        min_val = data.min()
        max_val = data.max()

        stats_text = (f"Median: {median:.4f} | IQR: {iqr:.4f}\n"
                      f"Range: [{min_val:.4f}, {max_val:.4f}]")

        # --- Plot 1: Boxplot ---
        plt.figure(figsize=(10, 4))
        # Boxplot shows Median, IQR (box), and Outliers (points beyond whiskers) by default
        sns.boxplot(x=data, color="skyblue", flierprops={"marker": "x", "markerfacecolor": "red"})
        plt.title(f"Centroid (GC) Analysis (Boxplot)\n{stats_text}")
        plt.xlabel("Cosine Similarity(Centroid(GC), Original)")
        plt.tight_layout()
        plt.xlim(0.8, 1.05)
        plt.savefig(output_dir / "exp2_centroid_boxplot.png")
        plt.close()
        print(f"Generated Exp 2 Boxplot: {output_dir / 'exp2_centroid_boxplot.png'}")
    else:
        print(f"Skipping Exp 2: {csv_exp1} not found.")

    # -------------------------------------------------------------------------
    # 3. Ablation Study (Exp 3)
    # -------------------------------------------------------------------------
    csv_exp3 = input_dir / "exp3_ablation.csv"
    if csv_exp3.exists():
        df3 = pd.read_csv(csv_exp3)
        # Melt
        df3_melt = pd.melt(df3, var_name="Strategy", value_name="Similarity(Single TTA, Original)")
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Strategy", y="Similarity(Single TTA, Original)", data=df3_melt, hue="Strategy", palette="viridis", legend=False)

        # 為每個 Strategy 類別畫一條垂直虛線 (axvline)
        # 獲取所有類別的數量
        num_strategies = len(df3_melt['Strategy'].unique())
        for i in range(num_strategies):
            # x=i 即為每個類別的中心位置
            # zorder=0 確保虛線在 Boxplot 後方，alpha 設置透明度避免太刺眼
            plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3, linewidth=1, zorder=0)

        plt.title("Ablation Study: TTA Strategy Robustness (Similarity to Original)")
        plt.xticks(rotation=45)
        plt.ylim(0.0, 1.05)
        plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "exp3_ablation.png")
        plt.close()
        print(f"Generated Exp 3 plot: {output_dir / 'exp3_ablation.png'}")
    else:
        print(f"Skipping Exp 3: {csv_exp3} not found.")

    # -------------------------------------------------------------------------
    # 4. Cohesion Analysis (Exp 4)
    # -------------------------------------------------------------------------
    csv_exp4 = input_dir / "exp4_cohesion.csv"
    if csv_exp4.exists():
        df4 = pd.read_csv(csv_exp4)
        df4_melt = pd.melt(df4, var_name="Strategy", value_name="Similarity_to_GC")
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Strategy", y="Similarity_to_GC", data=df4_melt, hue="Strategy", palette="magma", legend=False)

        # 為每個 Strategy 類別畫一條垂直虛線 (axvline)
        # 獲取所有類別的數量
        num_strategies = len(df3_melt['Strategy'].unique())
        for i in range(num_strategies):
            # x=i 即為每個類別的中心位置
            # zorder=0 確保虛線在 Boxplot 後方，alpha 設置透明度避免太刺眼
            plt.axvline(x=i, color='gray', linestyle='--', alpha=0.3, linewidth=1, zorder=0)

        plt.title("Cohesion Analysis: Distance from Group Consensus (GC)")
        plt.ylabel("Similarity(Single TTA, Centroid(GC))")
        plt.xticks(rotation=45)
        plt.ylim(0.0, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / "exp4_cohesion.png")
        plt.close()
        print(f"Generated Exp 4 plot: {output_dir / 'exp4_cohesion.png'}")
    else:
        print(f"Skipping Exp 4: {csv_exp4} not found.")

if __name__ == "__main__":
    main()

# uv run python src/model/ablation/visualize_results.py --input_dir results/ablation
