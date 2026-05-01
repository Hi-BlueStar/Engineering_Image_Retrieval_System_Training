import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def set_academic_style():
    """
    Configures Matplotlib to produce high-quality, XeLaTeX-compatible academic plots.
    """
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif", "serif"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
        "figure.figsize": (8, 6),
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05
    })
    
    sns.set_style("white")
    plt.rcParams["font.family"] = "serif"

def generate_dummy_data(num_samples=10000):
    """
    Generates synthetic similarity score data for testing.
    """
    np.random.seed(42)
    
    tp_scores = np.random.normal(loc=0.85, scale=0.08, size=int(num_samples * 0.1))
    tp_scores = np.clip(tp_scores, 0, 1.0)
    
    fp_scores = np.random.normal(loc=0.35, scale=0.15, size=int(num_samples * 0.9))
    fp_scores = np.clip(fp_scores, 0, 1.0)
    
    df_tp = pd.DataFrame({'score': tp_scores, 'label': 'True Positive'})
    df_fp = pd.DataFrame({'score': fp_scores, 'label': 'False Positive'})
    
    df = pd.concat([df_tp, df_fp], ignore_index=True)
    return df

def plot_retrieval_analysis(df, score_col, label_col, output_path, title=None):
    """
    Plots a Threshold-Oriented Distribution Chart:
    Top panel: Step Histogram (Density)
    Bottom panel: Boxplot
    """
    set_academic_style()
    
    # Identify labels and assign colors
    unique_labels = df[label_col].unique()
    custom_palette = {}
    tp_label = None
    fp_label = None
    
    for lbl in unique_labels:
        lbl_str = str(lbl).lower()
        if 'tp' in lbl_str or 'true positive' in lbl_str or lbl == 1:
            custom_palette[lbl] = "#0033a0" # Navy Blue
            tp_label = lbl
        elif 'fp' in lbl_str or 'false positive' in lbl_str or lbl == 0:
            custom_palette[lbl] = "#c8102e" # Brick Red
            fp_label = lbl
        else:
            custom_palette[lbl] = "#555555"
            
    # Create Figure with GridSpec (Height ratio 4:1)
    fig = plt.figure()
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
    
    ax_hist = fig.add_subplot(gs[0])
    ax_box = fig.add_subplot(gs[1], sharex=ax_hist)
    
    # 1. Step Histogram (Density)
    sns.histplot(
        data=df,
        x=score_col,
        hue=label_col,
        element="step",
        stat="density",
        common_norm=False,
        palette=custom_palette,
        alpha=0.4,
        linewidth=2,
        ax=ax_hist
    )
    
    # 2. Mean Lines & Contrastive Margin
    if tp_label and fp_label:
        tp_mean = df[df[label_col] == tp_label][score_col].mean()
        fp_mean = df[df[label_col] == fp_label][score_col].mean()
        
        ax_hist.axvline(tp_mean, color=custom_palette[tp_label], linestyle='--', linewidth=2, alpha=0.8)
        ax_hist.axvline(fp_mean, color=custom_palette[fp_label], linestyle='--', linewidth=2, alpha=0.8)
        
        margin = tp_mean - fp_mean
        ax_hist.text(
            0.5, 0.95, f"Contrastive Margin: {margin:.4f}\n(Mean TP: {tp_mean:.3f}, Mean FP: {fp_mean:.3f})", 
            transform=ax_hist.transAxes, 
            ha='center', va='top', 
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="lightgray", alpha=0.9),
            fontsize=12
        )
        
    # 3. Horizontal Boxplot
    sns.boxplot(
        data=df,
        x=score_col,
        y=label_col,
        hue=label_col,
        palette=custom_palette,
        legend=False,
        ax=ax_box,
        width=0.5,
        fliersize=3,
        linewidth=1.5
    )
    
    # Adjustments
    ax_hist.set_xlabel("")
    ax_hist.set_ylabel("Density")
    ax_box.set_xlabel("Similarity Score")
    ax_box.set_ylabel("")
    
    if title:
        ax_hist.set_title(title)
        
    # STRICT bounds
    ax_hist.set_xlim(0.0, 1.0)
    
    # Clean spines & ticks
    sns.despine(ax=ax_hist, bottom=True)
    sns.despine(ax=ax_box, left=True)
    
    plt.setp(ax_hist.get_xticklabels(), visible=False)
    ax_hist.tick_params(axis='x', which='both', length=0)
    
    ax_hist.grid(axis='y', linestyle='--', alpha=0.5)
    ax_box.grid(axis='x', linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, format='pdf', transparent=False)
    print(f"Plot saved successfully to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Academic Retrieval Analysis Plot (Redesigned)")
    parser.add_argument("--csv", type=str, help="Path to input CSV file containing retrieval results.")
    parser.add_argument("--output", type=str, default="retrieval_analysis.pdf", help="Path to save the output PDF plot.")
    parser.add_argument("--score_col", type=str, default="score", help="Column name for similarity score.")
    parser.add_argument("--label_col", type=str, default="label", help="Column name for True/False Positive label.")
    parser.add_argument("--title", type=str, default="Distribution of Similarity Scores", help="Plot title (optional).")
    parser.add_argument("--dummy", action="store_true", help="Generate and plot dummy data if no CSV is provided.")
    
    args = parser.parse_args()
    
    if args.csv and os.path.exists(args.csv):
        print(f"Loading data from {args.csv}...")
        df = pd.read_csv(args.csv)
        if args.score_col not in df.columns or args.label_col not in df.columns:
            raise ValueError(f"CSV must contain '{args.score_col}' and '{args.label_col}' columns.")
    elif args.dummy or not args.csv:
        print("No CSV provided or --dummy flag set. Generating synthetic dummy data...")
        df = generate_dummy_data()
        args.score_col = "score"
        args.label_col = "label"
    else:
        raise FileNotFoundError(f"Input file not found: {args.csv}")
        
    plot_retrieval_analysis(
        df=df,
        score_col=args.score_col,
        label_col=args.label_col,
        output_path=args.output,
        title=args.title
    )

if __name__ == "__main__":
    main()
