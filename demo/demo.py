import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from PIL import Image

# ChromaDB
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# --- Rich 終端美化模組 ---
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text
from rich.markup import escape
from rich import box

# 初始化 Rich Console
console = Console()

# --- 設定參數 ---
COLLECTION_NAME = "engineering_components_v1"
DB_PATH = "./chroma_db_store"
QUERY_FOLDER = "demo/image_to_be_searched"  # 來源圖片資料夾 (要搜尋的圖)
OUTPUT_FOLDER = "demo/search_results_viz"   # 結果圖表存檔位置

# 指定資料庫圖片的根目錄
# 程式將會依照這個路徑去尋找圖片： ROOT/{類別}/{料號}/{料號}_original.png
IMAGE_SOURCE_ROOT = "results/batch2/engineering_images_100dpi_2" 

N_DISPLAY = 10  # 最終要顯示幾張不重複的結果
N_FETCH = 30   # 從資料庫撈取幾張 (建議設比 N_DISPLAY 大，以防過濾重複後數量不足)

# --- 計時裝飾器類別 ---
class PhaseTimer:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.duration = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = (time.perf_counter() - self.start_time) * 1000 # 轉為毫秒

# --- 設定 Matplotlib 中文字型 ---
def set_chinese_font():
    """
    優先嘗試載入專案目錄下的字型檔，若無則嘗試系統字型。
    """
    # 1. 指定您下載的字型檔路徑 (請確保檔案存在於該處)
    # 建議將字型檔放在專案目錄下，例如: ./fonts/NotoSansTC-Bold.otf
    custom_font_path = "./src/NotoSansTC-VariableFont_wght.ttf" 
    
    if os.path.exists(custom_font_path):
        # 如果找到檔案，直接註冊並強制使用
        fm.fontManager.addfont(custom_font_path)
        # 取得該字型的內部名稱
        font_prop = fm.FontProperties(fname=custom_font_path)
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        return

    # 2. 如果沒檔案，嘗試尋找系統已安裝的字型
    font_candidates = [
        'Noto Sans TC', 'Noto Sans CJK TC', 'Microsoft JhengHei', 'PingFang TC', 
        'Heiti TC', 'SimHei', 'WenQuanYi Zen Hei', 'AR PL UMing TW'
    ]
    
    current_fonts = plt.rcParams.get('font.sans-serif', [])
    new_font_list = font_candidates + current_fonts if isinstance(current_fonts, list) else font_candidates
    plt.rcParams['font.sans-serif'] = new_font_list
    plt.rcParams['axes.unicode_minus'] = False
    
    # [檢查] 列印目前 Matplotlib 實際抓到的第一個字型，協助除錯
    # 如果這裡印出 'DejaVu Sans'，代表系統裡真的完全沒中文
    print(f"[系統檢查] 目前 Matplotlib 預設字型排序前三名: {plt.rcParams['font.sans-serif'][:3]}")

def init_db():
    with console.status("[bold green]正在初始化 ChromaDB 資料庫...", spinner="dots"):
        client = chromadb.PersistentClient(path=DB_PATH)
        embedding_func = OpenCLIPEmbeddingFunction()
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func,
            data_loader=ImageLoader()
        )
    console.print(f"[bold cyan]✔ 資料庫連線成功:[/bold cyan] {COLLECTION_NAME}")
    return collection

def construct_image_path(root_path, category, part_id):
    """
    依照指定格式重組圖片路徑
    格式: root/類別/料號/料號_original.png
    """
    if not category or not part_id:
        return None
    
    # 組合路徑
    filename = f"{part_id}_original.png"
    full_path = os.path.join(root_path, category, part_id, filename)
    return full_path

def visualize_and_save(query_img_path, query_image_obj, unique_results, save_path):
    """
    視覺化邏輯 
    1. 改為 Grid 佈局：每 4 張圖片一排，多列排列。
    2. 包含原始圖片(1張) + 檢索結果(N張)。
    3. 先儲存圖片，再顯示視窗，且不自動關閉視窗。
    """
    # 計算總圖片數量 (原始圖 + 結果圖)
    total_images = 1 + len(unique_results)
    cols = 4
    rows = math.ceil(total_images / cols)
    
    # 關閉互動視窗彈出，改為純後台繪圖以加快速度 (若需顯示可最後再開)
    # 設定畫布大小 (寬度固定，高度隨列數增加)
    plt.ioff() 
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    
    # 將 axes 展平為一維陣列，方便迴圈操作 (處理 1 row 或多 row 的情況)
    if isinstance(axes, np.ndarray):
        ax_list = axes.flatten()
    else:
        ax_list = [axes] 

    # --- 迴圈繪製每一格 ---
    for i in range(len(ax_list)):
        ax = ax_list[i]
        
        # 若格子索引超過圖片總數，則隱藏該格
        if i >= total_images:
            ax.axis('off')
            continue
            
        # === 第 1 張 (Index 0)：原始檢索圖像 ===
        if i == 0:
            ax.imshow(query_image_obj)
            title_text = f"【原始檢索圖像】\n{os.path.basename(query_img_path)}"
            # [中文顯示] 確保字體設定正確
            ax.set_title(title_text, color='red', fontsize=12, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            res_idx = i - 1
            part_id, category, sim_score, img_path = unique_results[res_idx]
            try:
                if img_path and os.path.exists(img_path):
                    res_img = Image.open(img_path)
                    ax.imshow(res_img)
                else:
                    ax.text(0.5, 0.5, "Image Not Found", ha='center', va='center', color='red')
                    ax.set_facecolor('#f0f0f0')
            except Exception:
                ax.text(0.5, 0.5, "Error", ha='center', va='center')

            info_text = f"Rank {res_idx+1}\n相似度: {sim_score:.3f}\n料號: {part_id}\n類別: {category}"
            ax.set_title(info_text, color='blue', fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig) # 繪圖完畢後關閉 figure 釋放記憶體

def print_performance_table(filename, times, result_count):
    """
    [新增] 繪製詳細的效能分析表格
    """
    # [修正點 1] 先將檔名轉義，防止檔名中有 [] 導致報錯
    safe_filename = escape(filename)
    
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("階段 (Stage)", style="cyan")
    table.add_column("耗時 (ms)", justify="right", style="green")
    table.add_column("佔比", justify="right", style="yellow")
    
    total_time = sum(times.values())
    
    # 加入各階段數據
    for stage, t in times.items():
        percent = (t / total_time) * 100 if total_time > 0 else 0
        table.add_row(stage, f"{t:.2f} ms", f"{percent:.1f}%")
        
    table.add_section()
    table.add_row("[bold]總計時間[/bold]", f"[bold white]{total_time:.2f} ms[/]", "")
    
    panel = Panel(
        table,
        # [修正點 2] 這裡使用 safe_filename
        title=f"[bold white]效能分析: {safe_filename}[/]",
        subtitle=f"[dim]找到 {result_count} 筆結果[/dim]",
        border_style="blue",
        expand=False
    )
    console.print(panel)

def process_folder(collection, input_folder, output_folder):
    if not os.path.exists(input_folder):
        console.print(f"[bold red]錯誤: 輸入資料夾不存在 {input_folder}[/bold red]")
        return
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    
    console.print(f"\n[bold yellow]🚀 準備處理 {len(files)} 張圖片...[/bold yellow]\n")

    # 使用 Rich 進度條
    with Progress(
        SpinnerColumn("earth"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None, style="cyan", complete_style="bold blue"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        
        main_task = progress.add_task("總體進度", total=len(files))

        for idx, filename in enumerate(files):
            progress.update(main_task, description=f"處理中: [bold white]{filename}[/bold white]")
            
            file_path = os.path.join(input_folder, filename)
            save_name = f"Result_{os.path.splitext(filename)[0]}.png"
            save_path = os.path.join(output_folder, save_name)
            
            times = {}
            unique_results = []

            try:
                # --- 1. 讀取圖片 ---
                with PhaseTimer("I/O 讀取") as t:
                    pil_image = Image.open(file_path).convert("RGB")
                    image_array = np.array(pil_image)
                times["讀取圖片"] = t.duration

                # --- 2. 向量搜尋 ---
                with PhaseTimer("AI 檢索") as t:
                    results = collection.query(
                        query_images=[image_array],
                        n_results=N_FETCH, 
                        include=['metadatas', 'distances']
                    )
                times["向量搜尋"] = t.duration

                # --- 3. 邏輯過濾 ---
                with PhaseTimer("資料處理") as t:
                    ids = results['ids'][0]
                    dists = results['distances'][0]
                    metas = results['metadatas'][0]
                    seen_part_ids = set()
                    
                    for id_val, dist, meta in zip(ids, dists, metas):
                        part_id = meta.get('part_id', 'Unknown')
                        category = meta.get('category', 'Unknown')
                        
                        if part_id in seen_part_ids:
                            continue
                        seen_part_ids.add(part_id)
                        
                        new_image_path = construct_image_path(IMAGE_SOURCE_ROOT, category, part_id)
                        similarity = max(0, 1 - dist)
                        unique_results.append((part_id, category, similarity, new_image_path))
                        
                        if len(unique_results) >= N_DISPLAY:
                            break
                times["邏輯過濾"] = t.duration

                # --- 4. 繪圖存檔 ---
                with PhaseTimer("視覺渲染") as t:
                    visualize_and_save(file_path, pil_image, unique_results, save_path)
                times["繪圖存檔"] = t.duration

                # 更新進度並顯示該張圖的詳細數據
                progress.advance(main_task)
                
                # 在進度條下方印出詳細表格 (不在進度條內，避免閃爍)
                safe_filename = escape(filename)
                console.print(f"✅ [bold green]完成:[/bold green] {safe_filename} -> {save_path}")
                print_performance_table(filename, times, len(unique_results))
                
            except Exception as e:
                # [修改點] 使用 escape() 包住 filename 和 str(e)
                # 這樣就算錯誤訊息裡有 [ ] 也不會導致報錯
                err_msg = escape(str(e))
                safe_filename = escape(filename)
                
                console.print(f"[bold red]❌ 處理失敗 {safe_filename}:[/bold red] {err_msg}")
                
                # 如果你想看完整的 traceback，建議保留這行
                import traceback
                traceback.print_exc()

# --- 主程式 ---
if __name__ == "__main__":
    # 酷炫標題
    title_text = """
    ██████╗ ██╗   ██╗    ███████╗███████╗██████╗ ██████╗  ██████╗██╗  ██╗
    ██╔══██╗╚██╗ ██╔╝    ██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║
    ██████╔╝ ╚████╔╝     ███████╗█████╗  ██████╔╝██████╔╝██║     ███████║
    ██╔═══╝   ╚██╔╝      ╚════██║██╔══╝  ██╔══██╗██╔══██╗██║     ██╔══██║
    ██║        ██║       ███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║
    ╚═╝        ╚═╝       ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
    """
    console.print(Panel(Text(title_text, justify="center", style="bold magenta"), border_style="cyan"))
    console.print("[bold yellow]正在啟動工程圖檢索系統 v0.1...[/bold yellow]\n")

    set_chinese_font()
    
    try:
        collection = init_db()
    except Exception as e:
        console.print(f"[bold red]資料庫連線失敗:[/bold red] {e}")
        exit()

    process_folder(collection, QUERY_FOLDER, OUTPUT_FOLDER)
    
    console.print("\n[bold green blink]✨ 所有任務執行完畢！ ✨[/bold green blink]")