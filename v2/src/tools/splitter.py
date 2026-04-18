"""資料集分割工具模組 (Dataset Splitter)。

負責抽樣並產生 train/val 的清單索引 (Manifest, CSV 格式)，取代無效率的實體檔案複製。
"""
import random
import pandas as pd
from collections import defaultdict
from pathlib import Path


class DatasetSplitter:
    def __init__(self, source_root: str, output_root: str, split_ratio: float = 0.8):
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.split_ratio = split_ratio
        self.structure_map = defaultdict(list)

    def scan(self) -> int:
        if not self.source_root.exists():
            print(f"[Warning] 分割來源目錄不存在: {self.source_root}")
            return 0
        
        total_files = 0
        current_class = ""
        for item in self.source_root.rglob("*.png"):
            # 假設結構: class_name / instance_name / {stem}_merged.png 等
            rel_parts = item.relative_to(self.source_root).parts
            if len(rel_parts) >= 2:
                class_name = rel_parts[0]
                instance_name = rel_parts[1]
                instance_dir = self.source_root / class_name / instance_name
                
                # 要確保不重複加入
                if instance_dir not in self.structure_map[class_name]:
                    self.structure_map[class_name].append(instance_dir)
                total_files += 1
                
        # 修正：直接掃描只會抓到底層，原邏輯是第一層 class，第二層 instance
        self.structure_map.clear()
        total_files = 0
        for class_dir in self.source_root.iterdir():
            if class_dir.is_dir():
                for inst_dir in class_dir.iterdir():
                    if inst_dir.is_dir():
                        self.structure_map[class_dir.name].append(inst_dir)
                        total_files += sum(1 for _ in inst_dir.rglob("*.png"))
        
        return total_files

    def split_run(self, run_idx: int, seed: int):
        run_name = f"Run_{run_idx:02d}_Seed_{seed}"
        out_dir = self.output_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        train_csv = out_dir / "train_manifest.csv"
        val_csv = out_dir / "val_manifest.csv"
        
        if train_csv.exists() and val_csv.exists():
            print(f"⏭ {run_name} Manifest 已存在，跳過分割。")
            return out_dir
            
        print(f"📂 開始產生 {run_name} 的分割索引 (Seed: {seed})...")
        random.seed(seed)
        
        train_list = []
        val_list = []
        
        def _collect_instance(inst_dir: Path, target_list: list):
            large_comp = inst_dir / "large_components"
            if large_comp.exists() and large_comp.is_dir():
                for p in large_comp.glob("*.png"):
                    # 儲存絕對路徑或相對於專案的相對路徑
                    target_list.append(str(p.resolve()))
        
        for cls_name, insts in self.structure_map.items():
            shuffled = list(insts)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * self.split_ratio)
            
            for inst in shuffled[:split_idx]:
                _collect_instance(inst, train_list)
            for inst in shuffled[split_idx:]:
                _collect_instance(inst, val_list)
                
        # 寫入 CSV
        pd.DataFrame({"filepath": train_list}).to_csv(train_csv, index=False)
        pd.DataFrame({"filepath": val_list}).to_csv(val_csv, index=False)

        print(f"   ✔ 分割完成! Train components: {len(train_list)}, Val components: {len(val_list)}")
        return out_dir
