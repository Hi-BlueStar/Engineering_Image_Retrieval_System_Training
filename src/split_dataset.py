import os
import shutil
import random
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

# Rich 相關引用
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    TaskProgressColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.box import ROUNDED
from rich.markup import escape
from rich.style import Style

class RichDatasetSplitter:
    def __init__(self, source_root: str, output_root: str, split_ratio: float = 0.8):
        self.source_root = Path(source_root)
        self.output_root = Path(output_root)
        self.split_ratio = split_ratio
        
        # 初始化 Rich Console
        self.console = Console()
        
        # 資料結構: { 'ClassName': [Path(Instance_1), Path(Instance_2)...] }
        self.structure_map = defaultdict(list)
        
        # 統計數據 (用於最後生成報表)
        self.stats = []

    def scan_dataset_structure(self):
        """
        第一階段：使用 Spinner 顯示掃描過程
        """
        if not self.source_root.exists():
            # 使用 escape 處理路徑，並用紅色粗體顯示錯誤
            error_msg = f"[bold red]找不到來源資料夾: {escape(str(self.source_root))}[/]"
            self.console.print(error_msg)
            raise FileNotFoundError(self.source_root)

        # 1. 初始化階段：使用 console.status 顯示轉圈動畫
        with self.console.status("[bold green]正在掃描資料夾結構與計算檔案數量...", spinner="dots"):
            count = 0
            total_files = 0
            
            # 遍歷第一層 (類別層級)
            for class_dir in self.source_root.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    # 遍歷第二層 (工件層級)
                    for instance_dir in class_dir.iterdir():
                        if instance_dir.is_dir():
                            self.structure_map[class_name].append(instance_dir)
                            count += 1
                            # 預先計算該 Instance 下的所有 PNG 數量 (為了進度條準確度)
                            total_files += self._count_png_recursive(instance_dir)
            
            time.sleep(0.5) # 稍微暫停讓使用者看到完成狀態
            
        self.console.print(f"[green]✔[/] 掃描完成。共發現 [bold cyan]{len(self.structure_map)}[/] 個類別，[bold cyan]{count}[/] 個工件實例。")
        self.console.print(f"   預計處理圖片總數: [bold yellow]{total_files}[/]\n")
        return total_files

    def _count_png_recursive(self, path: Path) -> int:
        """輔助函式：計算路徑下所有 PNG 數量 (包含子目錄)"""
        count = 0
        for p in path.rglob("*.png"):
            if p.is_file():
                count += 1
        return count

    def _safe_copy(self, src_file: Path, dst_dir: Path, dst_filename: str = None) -> bool:
        """安全的複製檔案，若檔名重複則自動更名。可指定目標檔名。"""
        try:
            # 如果有指定新檔名就用指定的，否則用原檔名
            target_name = dst_filename if dst_filename else src_file.name
            dst_file = dst_dir / target_name
            
            if dst_file.exists():
                # 注意：這裡要針對 target_name 取 stem 和 suffix
                path_obj = Path(target_name)
                stem = path_obj.stem
                suffix = path_obj.suffix
                counter = 1
                while dst_file.exists():
                    dst_file = dst_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.copy2(src_file, dst_file)
            return True
        except Exception as e:
            # 4. 錯誤處理：捕捉個別檔案錯誤但不中斷整個流程
            self.console.print(f"[bold red]複製失敗:[/ {escape(str(src_file))} -> {escape(str(e))}")
            return False

    def copy_files_batch(self, instance_list: List[Path], dataset_type: str, seed_dir: Path, progress, task_id) -> Tuple[int, int]:
        """
        執行實際 I/O，並更新外部傳入的 progress task
        """
        inst_out_dir = seed_dir / "Instance_Dataset" / dataset_type
        comp_out_dir = seed_dir / "Component_Dataset" / dataset_type

        inst_out_dir.mkdir(parents=True, exist_ok=True)
        comp_out_dir.mkdir(parents=True, exist_ok=True)

        count_inst = 0
        count_comp = 0

        for instance_path in instance_list:
            # A. Instance Level
            for item in instance_path.iterdir():
                if item.is_file() and item.suffix.lower() == '.png':
                    self._safe_copy(item, inst_out_dir)
                    count_inst += 1
                    progress.advance(task_id) # 更新進度條
            
            # B. Component Level
            large_comp_dir = instance_path / "large_components"
            if large_comp_dir.exists() and large_comp_dir.is_dir():
                
                # 取得 ID (即 instance_path 的資料夾名稱，例如 "SY3CHH03-10060102000")
                instance_id = instance_path.name
                
                for item in large_comp_dir.iterdir():
                    if item.is_file() and item.suffix.lower() == '.png':
                        # 組合新檔名: ID + "_" + 原檔名
                        new_filename = f"{instance_id}_{item.name}"
                        
                        # 傳入新檔名
                        self._safe_copy(item, comp_out_dir, dst_filename=new_filename)
                        count_comp += 1
                        progress.advance(task_id) # 更新進度條
        
        return count_inst, count_comp

    def run_repeated_splits(self, start_seed: int, repeats: int):
        """
        主流程：包含進度條與統計
        """
        try:
            total_files_in_dataset = self.scan_dataset_structure()
            
            # 2. 執行過程：使用 rich.progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40, style="blue", complete_style="green"),
                TaskProgressColumn(),      # 百分比 
                MofNCompleteColumn(),      # 10/100 檔案數
                TimeElapsedColumn(),       # 已耗時
                console=self.console,
                transient=False            # 完成後保留進度條
            ) as progress:
                
                overall_task = progress.add_task(f"[bold]總進度 ({repeats} Runs)", total=repeats)
                
                for i in range(repeats):
                    seed = start_seed + i
                    run_name = f"Run_{i+1:02d}"
                    
                    # 開始計時
                    start_time = time.perf_counter()
                    
                    # 設定隨機
                    random.seed(seed)
                    current_output_dir = self.output_root / f"{run_name}_Seed_{seed}"
                    
                    # 建立該 Run 的檔案處理進度條 (Total = 總圖片數 * 2 因為 copy 兩次? 不，這裡是拆分，總數即為資料集總數)
                    # 注意：分層抽樣後，train+val 的總圖數應該等於 total_files_in_dataset
                    file_task = progress.add_task(f"  └─ {run_name} Processing...", total=total_files_in_dataset)
                    
                    run_train_count = 0
                    run_val_count = 0
                    
                    # 執行分層拆分
                    for class_name, instances in self.structure_map.items():
                        instances_shuffled = instances[:]
                        random.shuffle(instances_shuffled)
                        
                        split_idx = int(len(instances_shuffled) * self.split_ratio)
                        train_instances = instances_shuffled[:split_idx]
                        val_instances = instances_shuffled[split_idx:]
                        
                        # 複製並更新進度
                        t_inst, t_comp = self.copy_files_batch(train_instances, 'train', current_output_dir, progress, file_task)
                        v_inst, v_comp = self.copy_files_batch(val_instances, 'val', current_output_dir, progress, file_task)
                        
                        run_train_count += (t_inst + t_comp)
                        run_val_count += (v_inst + v_comp)

                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    # 記錄統計資料
                    self.stats.append({
                        "run": run_name,
                        "seed": seed,
                        "train_files": run_train_count,
                        "val_files": run_val_count,
                        "duration": duration
                    })
                    
                    # 更新總進度與移除子任務
                    progress.advance(overall_task)
                    progress.update(file_task, visible=False) # 隱藏完成的子任務

            # 3. 結果展示：生成 Panel 與 Table
            self._show_summary_report()

        except Exception as e:
            # 4. 錯誤處理：紅色粗體 + escape
            self.console.print(f"\n[bold red]程式執行發生嚴重錯誤:[/]\n{escape(str(e))}")

    def _show_summary_report(self):
        """生成最終美化報表"""
        table = Table(title="Dataset Split Statistics", box=ROUNDED, show_lines=True)
        
        table.add_column("Run ID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Seed", justify="center", style="magenta")
        table.add_column("Train Files", justify="right", style="green")
        table.add_column("Val Files", justify="right", style="yellow")
        table.add_column("Duration (s)", justify="right", style="white")

        total_time = 0
        for stat in self.stats:
            table.add_row(
                stat["run"],
                str(stat["seed"]),
                f"{stat['train_files']:,}",
                f"{stat['val_files']:,}",
                f"{stat['duration']:.2f}"
            )
            total_time += stat['duration']

        # 建立 Panel
        summary_panel = Panel(
            table,
            title="[bold]✅ Execution Complete[/]",
            subtitle=f"Total Time: {total_time:.2f}s",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
        self.console.print("\n")
        self.console.print(summary_panel)

# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    # 假定參數
    SOURCE_DIR = "results/batch2/engineering_images_100dpi_2"  # 您的來源路徑
    OUTPUT_DIR = "dataset"  # 輸出路徑
    BASE_SEED = 42  # 初始種子碼
    REPEAT_TIMES = 20  # 重複執行次數 (N)
    TRAIN_RATIO = 0.8  # 訓練集比例
    
    # 執行
    splitter = RichDatasetSplitter(SOURCE_DIR, OUTPUT_DIR, TRAIN_RATIO)
    splitter.run_repeated_splits(BASE_SEED, REPEAT_TIMES)
