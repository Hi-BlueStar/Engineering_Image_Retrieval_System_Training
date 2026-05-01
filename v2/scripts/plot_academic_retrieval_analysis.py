import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def set_academic_style():
    """
    Configures Matplotlib to produce high-quality, XeLaTeX-compatible academic plots.
    Features:
    - TrueType fonts for PDF/PS output (editable in Illustrator, no missing fonts in LaTeX).
    - Serif font family (Times-like) preferred for international papers.
    - Clean axes (minimalist spines).
    - High contrast, large legible fonts.
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
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05
    })
    
    # Use seaborn white style as a base for a clean look
    sns.set_style("white")
    # Re-apply font settings as seaborn might override them
    plt.rcParams["font.family"] = "serif"

def generate_dummy_data(num_samples=1000):
    """
    Generates synthetic similarity score data simulating True Positives and False Positives.
    """
    np.random.seed(42)
    
    # Simulate TP: higher mean similarity, narrower spread
    tp_scores = np.random.normal(loc=0.85, scale=0.1, size=int(num_samples * 0.4))
    tp_scores = np.clip(tp_scores, 0, 1.0)
    
    # Simulate FP: lower mean similarity, wider spread
    fp_scores = np.random.normal(loc=0.45, scale=0.2, size=int(num_samples * 0.6))
    fp_scores = np.clip(fp_scores, 0, 1.0)
    
    df_tp = pd.DataFrame({'score': tp_scores, 'label': 'True Positive'})
    df_fp = pd.DataFrame({'score': fp_scores, 'label': 'False Positive'})
    
    df = pd.concat([df_tp, df_fp], ignore_index=True)
    return df

def plot_retrieval_analysis(df, score_col, label_col, output_path, title=None):
    """
    Plots the KDE distribution and rug plot for similarity scores, distinguishing TP and FP.
    """
    set_academic_style()
    
    fig, ax = plt.subplots()
    
    # High-contrast, color-blind friendly palette
    # TP: Navy Blue, FP: Brick Red
    palette = {"True Positive": "#0033a0", "False Positive": "#c8102e"}
    
    # If custom labels exist in data, map colors dynamically, but prioritize TP/FP matching
    unique_labels = df[label_col].unique()
    custom_palette = {}
    for lbl in unique_labels:
        lbl_str = str(lbl).lower()
        if 'tp' in lbl_str or 'true positive' in lbl_str or lbl == 1:
            custom_palette[lbl] = "#0033a0"
        elif 'fp' in lbl_str or 'false positive' in lbl_str or lbl == 0:
            custom_palette[lbl] = "#c8102e"
        else:
            custom_palette[lbl] = "#555555" # fallback gray
    
    # 1. Plot KDE (Density)
    sns.kdeplot(
        data=df,
        x=score_col,
        hue=label_col,
        fill=True,
        alpha=0.4,
        linewidth=2,
        palette=custom_palette,
        ax=ax,
        warn_singular=False
    )
    
    # 2. Add Rug plot at the bottom to show actual data points
    sns.rugplot(
        data=df,
        x=score_col,
        hue=label_col,
        height=0.05,
        alpha=0.5,
        expand_margins=True,
        palette=custom_palette,
        ax=ax
    )
    
    # 3. Customizations for academic look
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
        
    ax.set_xlim(0.0, 1.0)
    
    # Clean up spines (remove top and right)
    sns.despine(ax=ax, offset=5, trim=False)
    
    # Add light horizontal grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Save the figure
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, format='pdf', transparent=False)
    print(f"Plot saved successfully to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Academic Retrieval Analysis Plot")
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
