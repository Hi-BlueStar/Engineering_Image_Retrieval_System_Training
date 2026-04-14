# Run Analysis
uv run python src/model/ablation/tta_gc_analysis.py \
    --checkpoint outputs/simsiam_exp_01_Run_01_Seed_42_20260130_105404/checkpoints/checkpoint_best.pth \
    --data_root data/吉輔提供資料Clean \
    --output_dir results/ablation
# Visualize Results
uv run python src/model/ablation/visualize_results.py --input_dir results/ablation