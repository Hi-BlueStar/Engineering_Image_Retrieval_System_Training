import os
import glob
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import numpy as np

def analyze_aspect_ratios(image_dir):
    print(f"Analyzing images in {image_dir}...")
    image_paths = glob.glob(os.path.join(image_dir, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(image_dir, "*.[pP][nN][gG]")) + \
                  glob.glob(os.path.join(image_dir, "*.[jJ][pP][eE][gG]"))
    
    data = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                ratio = w / h
                data.append({
                    'path': path,
                    'width': w,
                    'height': h,
                    'ratio': ratio
                })
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    return data

def plot_histogram(data):
    ratios = [d['ratio'] for d in data]
    plt.figure(figsize=(10, 6))
    plt.hist(ratios, bins=50, color='skyblue', edgecolor='black')
    plt.title("Aspect Ratio (Width / Height) Distribution")
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.75)
    plt.show()

class ImageRotator:
    def __init__(self, images_to_process):
        self.images = images_to_process
        self.index = 0
        self.current_img = None
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Add buttons
        ax_prev = plt.axes([0.1, 0.05, 0.15, 0.075])
        ax_ccw = plt.axes([0.3, 0.05, 0.15, 0.075])
        ax_cw = plt.axes([0.5, 0.05, 0.15, 0.075])
        ax_next = plt.axes([0.75, 0.05, 0.15, 0.075])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_ccw = Button(ax_ccw, 'Rotate 90 CCW')
        self.btn_cw = Button(ax_cw, 'Rotate 90 CW')
        self.btn_next = Button(ax_next, 'Next / Save')
        
        self.btn_prev.on_clicked(self.prev)
        self.btn_ccw.on_clicked(self.rotate_ccw)
        self.btn_cw.on_clicked(self.rotate_cw)
        self.btn_next.on_clicked(self.next)
        
        self.show_image()
        plt.show()

    def show_image(self):
        if 0 <= self.index < len(self.images):
            item = self.images[self.index]
            self.current_path = item['path']
            self.current_img = Image.open(self.current_path)
            self.ax.clear()
            self.ax.imshow(self.current_img)
            self.ax.set_title(f"Image {self.index + 1}/{len(self.images)}: {os.path.basename(self.current_path)}\nRatio: {item['ratio']:.2f}")
            self.ax.axis('off')
            plt.draw()
        else:
            print("No more images.")
            plt.close()

    def rotate_ccw(self, event):
        self.current_img = self.current_img.rotate(90, expand=True)
        self.ax.clear()
        self.ax.imshow(self.current_img)
        self.ax.axis('off')
        plt.draw()

    def rotate_cw(self, event):
        self.current_img = self.current_img.rotate(-90, expand=True)
        self.ax.clear()
        self.ax.imshow(self.current_img)
        self.ax.axis('off')
        plt.draw()

    def next(self, event):
        # Save changes if any (only if rotated)
        self.current_img.save(self.current_path)
        print(f"Saved {self.current_path}")
        self.index += 1
        self.show_image()

    def prev(self, event):
        self.index -= 1
        self.show_image()

def main():
    # Use absolute paths to avoid issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(base_dir, "data", "converted_images")
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found.")
        return

    # 1. Analyze
    results = analyze_aspect_ratios(data_dir)
    if not results:
        print("No images found.")
        return

    # 2. Plot
    plot_histogram(results)

    # 3. Ask for threshold
    try:
        print("\nReview the chart and decide the range of 'normal' aspect ratios.")
        min_ratio = float(input("Enter minimum aspect ratio threshold (e.g., 0.5): "))
        max_ratio = float(input("Enter maximum aspect ratio threshold (e.g., 2.0): "))
    except ValueError:
        print("Invalid input. Using default (0 to infinity).")
        min_ratio = 0
        max_ratio = float('inf')

    # 4. Filter
    out_of_range = [d for d in results if d['ratio'] < min_ratio or d['ratio'] > max_ratio]
    print(f"Found {len(out_of_range)} images out of range.")

    if not out_of_range:
        print("Everything is within range. Exiting.")
        return

    # 5. Interactive UI
    rotator = ImageRotator(out_of_range)

if __name__ == "__main__":
    main()
