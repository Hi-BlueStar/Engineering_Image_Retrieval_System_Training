#!/usr/bin/env python3
"""自動化執行 Baseline, Exp A, Exp B 實驗的腳本。"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="自動化執行所有 SimSiam 圖像檢索實驗")
    parser.add_argument("--scale", type=str, default="tsmall", choices=["tsmall", "tlarge"], help="實驗規模大小 (tsmall 或 tlarge)")
    parser.add_argument("--epochs", type=int, default=10, help="每個實驗的訓練 Epoch 數（預設小規模測試為 1）")
    parser.add_argument("--max-batches", type=int, default=128, help="每個 Epoch 的最大 Batch 數（預設小規模測試為 2）")
    parser.add_argument("--output-dir", type=str, default="outputs_v2", help="輸出目錄路徑")
    parser.add_argument("--eval-freq", type=int, default=1, help="評估頻率（Epochs）")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    train_script = project_root / "v2" / "train.py"
    
    experiments = [f"baseline_{args.scale}", f"exp_a_{args.scale}", f"exp_b_{args.scale}"]
    
    print("=" * 60)
    print("啟動自動化實驗管線")
    print(f"專案根目錄: {project_root}")
    print(f"訓練腳本: {train_script}")
    print(f"參數設定: epochs={args.epochs}, max_batches={args.max_batches}, output_dir={args.output_dir}")
    print("=" * 60)

    for exp in experiments:
        print(f"\n[開始執行實驗] => {exp}")
        
        # 建立執行指令，使用 sys.executable 以保持當前虛擬環境
        cmd = [
            sys.executable,
            str(train_script),
            f"experiment.output_dir={args.output_dir}",
            f"training.epochs={args.epochs}",
            f"training.max_batches={args.max_batches}",
            f"experiment.eval_freq={args.eval_freq}",
            f"experiment={exp}"
        ]
        
        print(f"執行指令: {' '.join(cmd)}")
        try:
            # 實時輸出子進程的 stdout/stderr
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                check=True,
                text=True
            )
            print(f"[實驗完成] => {exp} 執行成功！")
        except subprocess.CalledProcessError as err:
            print(f"[錯誤] => {exp} 執行失敗，錯誤碼: {err.returncode}", file=sys.stderr)
            sys.exit(err.returncode)
            
    print("\n" + "=" * 60)
    print("所有實驗執行完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
