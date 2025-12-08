import sys
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

if __name__ == "__main__":
    # 清除螢幕 (Windows/Unix 兼容)
    os.system('cls' if os.name == 'nt' else 'clear')
    
    banner = generate_cyberpunk_banner("工程圖檢索系統", width=180)
    
    print("\n" + " " * 5 + "SYSTEM INITIALIZATION SEQUENCE STARTED...")
    print("-" * 120)
    print(banner)
    print("-" * 120)
    # 模擬系統狀態列
    print(rgb_to_ansi(0, 255, 255, " [ CORE: ONLINE ] ") + 
          " " * 60 + 
          rgb_to_ansi(255, 191, 0, " [ SECURITY: HIGH ] "))