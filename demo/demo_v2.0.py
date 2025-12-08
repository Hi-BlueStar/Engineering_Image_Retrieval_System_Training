import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap

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
COLLECTION_NAME = "engineering_components_v1_demo"
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

# 用於 PIL 的字型載入 helper (因為 PIL 不像 Matplotlib 會自動抓系統字型)
def get_pil_font(size=20):
    # 優先嘗試專案目錄下的字型
    custom_font_path = "./src/NotoSansTC-VariableFont_wght.ttf"
    
    # 常見的系統中文字型路徑 (Linux/Windows/Mac)
    system_fonts = [
        "msjh.ttc", "msjh.ttf",  # Windows 微軟正黑體
        "NotoSansTC-Regular.otf", # Linux/Mac
        "/usr/share/fonts/opentype/noto/NotoSansTC-Bold.otf",
        "/System/Library/Fonts/PingFang.ttc" # Mac
    ]
    
    path_to_use = custom_font_path if os.path.exists(custom_font_path) else None
    
    if not path_to_use:
        # 簡單搜尋系統字型 (這裡僅作示範，實務上可更嚴謹)
        import platform
        sys_type = platform.system()
        if sys_type == "Windows":
             # 嘗試在 Windows Fonts 資料夾找
             for f in system_fonts:
                 p = os.path.join("C:\\Windows\\Fonts", f)
                 if os.path.exists(p):
                     path_to_use = p
                     break
    
    try:
        if path_to_use:
            return ImageFont.truetype(path_to_use, size)
        else:
            # 若真的找不到，回退到預設 (不支援中文)
            return ImageFont.load_default()
    except:
        return ImageFont.load_default()

def fit_text_to_box(draw, text, font, max_width, max_lines=3):
    """
    [輔助函式] 自動折行與截斷文字
    1. 根據 max_width 自動換行
    2. 超過 max_lines 則截斷並補上 "..."
    """
    # 估算單一字元平均寬度 (稍微寬容一點)
    avg_char_width = font.getlength("A") 
    # 估算一行能塞幾個字
    chars_per_line = int(max_width / avg_char_width)
    
    # 使用 textwrap 進行折行
    lines = textwrap.wrap(text, width=chars_per_line)
    
    # 如果行數過多，進行截斷
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "..." # 最後一行尾端加省略號
        
    return lines

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

# [重構] 使用 PIL 直接繪圖，取代 Matplotlib
def visualize_and_save(query_img_path, query_image_obj, unique_results, save_path):
    """
    [優化版] 視覺化邏輯
    1. 保持圖片原始比例 (Aspect Ratio Preserved)
    2. 文字自動折行與防溢出 (Text Wrapping)
    3. 純 PIL 繪製，極速渲染
    """
    # --- 參數設定 ---
    CELL_W, CELL_H = 370, 330    # 每個格子的圖片區域大小 (含留白)
    IMG_MAX_W, IMG_MAX_H = 360, 320 # 圖片實際最大限制 (略小於格子，製造邊框感)
    TEXT_H = 20                 # 底部文字區域高度 (增加高度以容納多行)
    PADDING = 5                 # 格子間距
    COLS = 4                     # 每排幾張
    
    BG_COLOR = (250, 250, 250)   # 背景色 (微灰，提升質感)
    CARD_COLOR = (255, 255, 255) # 卡片背景色 (純白)
    
    # 計算總圖片數與畫布大小
    total_images = 1 + len(unique_results)
    rows = math.ceil(total_images / COLS)
    
    canvas_w = COLS * (CELL_W + PADDING) + PADDING
    canvas_h = rows * (CELL_H + TEXT_H + PADDING) + PADDING
    
    # 建立大畫布
    canvas = Image.new('RGB', (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    
    # 載入字型
    font_title = get_pil_font(size=20) # 標題字
    font_meta = get_pil_font(size=16)  # 內文字
    
    # 準備迭代清單
    items = []
    # 1. 原始檢索圖
    q_name = os.path.basename(query_img_path)
    items.append({
        "img": query_image_obj, 
        "title": "【原始檢索圖像】",
        "detail": q_name,
        "is_query": True,
        "score": None
    })
    
    # 2. 檢索結果
    for i, (part_id, category, sim_score, img_path) in enumerate(unique_results):
        try:
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.new('RGB', (100, 100), (200, 200, 200)) # Placeholder
        except:
            img = Image.new('RGB', (100, 100), (200, 200, 200))
            
        items.append({
            "img": img,
            "title": f"Rank {i+1} (Sim: {sim_score:.3f})",
            "detail": f"PN: {part_id}\nCat: {category}",
            "is_query": False,
            "score": sim_score
        })

    # --- 繪製迴圈 ---
    for idx, item in enumerate(items):
        r, c = divmod(idx, COLS)
        
        # 計算目前格子的左上角座標
        cell_x = PADDING + c * (CELL_W + PADDING)
        cell_y = PADDING + r * (CELL_H + TEXT_H + PADDING)
        
        # 1. 繪製卡片背景 (白色圓角矩形概念，這裡用矩形代替)
        draw.rectangle(
            [cell_x, cell_y, cell_x + CELL_W, cell_y + CELL_H + TEXT_H], 
            fill=CARD_COLOR, outline=(220, 220, 220), width=1
        )
        
        # 2. 處理圖片 (保持比例縮放)
        # ImageOps.contain 會自動計算縮放比例，保持長寬比並塞入指定大小
        img_contain = ImageOps.contain(item["img"], (IMG_MAX_W, IMG_MAX_H), method=Image.Resampling.LANCZOS)
        
        # 計算居中位置
        img_x = cell_x + 5
        img_y = cell_y + 5
        
        canvas.paste(img_contain, (img_x, img_y))
        
        # 若是 Query 圖，畫紅框加強提示
        if item["is_query"]:
            draw.rectangle(
                [cell_x, cell_y, cell_x + CELL_W, cell_y + CELL_H + TEXT_H], 
                outline=(255, 0, 0), width=4
            )

        # 3. 繪製文字
        text_start_y = cell_y + IMG_MAX_H - 40  # 文字從圖片區下方開始
        text_x = cell_x + 10               # 左側留點邊距
        
        # (A) 標題 (Rank / Sim)
        draw.text((text_x, text_start_y), item["title"], fill=(0, 0, 200) if not item["is_query"] else (200, 0, 0), font=font_title)
        
        # (B) 詳細資訊 (料號 / 檔名) - 需處理過長
        detail_lines = fit_text_to_box(draw, item["detail"], font_meta, max_width=CELL_W - 20, max_lines=3)
        
        current_y = text_start_y + 30
        for line in detail_lines:
            draw.text((text_x, current_y), line, fill=(50, 50, 50), font=font_meta)
            current_y += 20 # 行距

    # --- 存檔 ---
    # 使用 optimize=True 壓縮 PNG 大小，加快 I/O 寫入
    canvas.save(save_path, format="PNG", optimize=True)

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






###############################################################################
##                                  酷炫標題                                  ##
###############################################################################

def get_gradient_color(start_rgb, end_rgb, factor):
    """
    計算兩個 RGB 顏色之間的線性插值。
    factor: 0.0 (start) ~ 1.0 (end)
    """
    r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * factor)
    g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * factor)
    b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * factor)
    return r, g, b

def rgb_to_ansi(r, g, b, text):
    """
    將文字包裹在 ANSI TrueColor 轉義序列中。
    """
    return f"\033[38;2;{r};{g};{b}m{text}\033[0m"

def generate_cyberpunk_banner(text, width=100):
    # 1. 定義顏色：青色 (Cyan) -> 琥珀色 (Amber)
    COLOR_CYAN = (0, 255, 255)   # 冷色調
    COLOR_AMBER = (255, 191, 0)  # 暖色調
    
    # 2. 字元密度表 (高解析度映射)
    ascii_chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

    # 3. 設定字體 (自動偵測作業系統)
    try:
        font_path = "./src/NotoSansTC-VariableFont_wght.ttf" 
            
        font_size = 80
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"錯誤：找不到字體文件於 {font_path}，請手動指定路徑。")
        return

    # 4. 繪製高解析度底圖
    temp_img = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    
    original_w, original_h = bbox[2], bbox[3]
    image = Image.new("L", (original_w, original_h + 10), 0) # 黑底白字
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill=255)

    # 5. 縮放 (保持工程圖比例)
    aspect_ratio = original_h / original_w
    new_height = int(aspect_ratio * width * 0.55)
    resized_image = image.resize((width, new_height), Image.Resampling.LANCZOS)
    
    pixels = np.array(resized_image)
    char_len = len(ascii_chars)
    
    # 6. 生成帶顏色的 ASCII Art
    output_lines = []
    
    for y, row in enumerate(pixels):
        line_str = ""
        for x, pixel_value in enumerate(row):
            # 取得對應的 ASCII 字元
            char_index = int(pixel_value / 255 * (char_len - 1))
            char = ascii_chars[char_index]
            
            # 只有當像素夠亮（有內容）時才上色，空白處保持原樣以節省運算並保持乾淨
            if pixel_value > 20: 
                # 計算水平漸層位置 (0.0 ~ 1.0)
                gradient_factor = x / width
                r, g, b = get_gradient_color(COLOR_CYAN, COLOR_AMBER, gradient_factor)
                line_str += rgb_to_ansi(r, g, b, char)
            else:
                line_str += " " # 空白
                
        output_lines.append(line_str)
        
    return "\n".join(output_lines)


# --- 主程式 ---
if __name__ == "__main__":
    # 酷炫標題
    banner = generate_cyberpunk_banner("工程圖檢索系統", width=200)
    
    print(banner)
    console.print("[bold yellow]正在啟動工程圖檢索系統 v0.1...[/bold yellow]\n")

    set_chinese_font()
    
    try:
        collection = init_db()
    except Exception as e:
        console.print(f"[bold red]資料庫連線失敗:[/bold red] {e}")
        exit()

    process_folder(collection, QUERY_FOLDER, OUTPUT_FOLDER)
    
    console.print("\n[bold green blink]✨ 所有任務執行完畢！ ✨[/bold green blink]")