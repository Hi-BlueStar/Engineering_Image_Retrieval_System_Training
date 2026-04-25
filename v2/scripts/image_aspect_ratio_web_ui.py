import os
import glob
import io
import base64
from flask import Flask, render_template_string, request, jsonify, send_from_directory
from PIL import Image
import matplotlib.pyplot as plt

app = Flask(__name__)

# Configuration
WORKSPACE_ROOT = "/home/master-user/Desktop/Engineering_Image_Retrieval_System_Training"
IMAGE_DIR = os.path.join(WORKSPACE_ROOT, "data", "converted_images")

# Global state to store analysis results
analysis_results = []

def analyze_images():
    global analysis_results
    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.[jJ][pP][gG]")) + \
                  glob.glob(os.path.join(IMAGE_DIR, "*.[pP][nN][gG]")) + \
                  glob.glob(os.path.join(IMAGE_DIR, "*.[jJ][pP][eE][gG]"))
    
    results = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
                ratio = w / h
                results.append({
                    'filename': os.path.basename(path),
                    'path': path,
                    'width': w,
                    'height': h,
                    'ratio': ratio
                })
        except Exception as e:
            print(f"Error processing {path}: {e}")
    analysis_results = results
    return results

def generate_histogram_base64(results):
    ratios = [d['ratio'] for d in results]
    plt.figure(figsize=(10, 5))
    plt.hist(ratios, bins=50, color='skyblue', edgecolor='black')
    plt.title("Aspect Ratio (Width / Height) Distribution")
    plt.xlabel("Aspect Ratio")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.75)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Aspect Ratio Manager</title>
    <style>
        body { font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f4f4f9; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1 { color: #333; }
        img.main-img { max-width: 100%; max-height: 500px; border: 1px solid #ddd; }
        .controls { margin-top: 20px; display: flex; gap: 10px; align-items: center; }
        button { padding: 10px 20px; cursor: pointer; border: none; border-radius: 4px; background: #007bff; color: white; }
        button:hover { background: #0056b3; }
        button.secondary { background: #6c757d; }
        .hidden { display: none; }
        .stats { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Image Aspect Ratio Manager</h1>
    
    <div id="setup-step" class="card">
        <h2>1. Analysis</h2>
        <button onclick="runAnalysis()">Run Analysis</button>
        <div id="analysis-output" class="hidden">
            <div style="margin-top: 20px;">
                <img id="histogram" style="width: 100%;" />
            </div>
            <div class="controls">
                <label>Min Ratio: <input type="number" id="min-ratio" value="0.5" step="0.1"></label>
                <label>Max Ratio: <input type="number" id="max-ratio" value="2.0" step="0.1"></label>
                <button onclick="startReview()">Start Review Out-of-Range Images</button>
            </div>
        </div>
    </div>

    <div id="review-step" class="card hidden">
        <h2>2. Reviewing <span id="current-index-display">0</span> / <span id="total-count-display">0</span></h2>
        <div class="stats" id="img-info"></div>
        <div style="text-align: center; margin-top: 10px;">
            <img id="review-img" class="main-img" src="" />
        </div>
        <div class="controls" style="justify-content: center;">
            <button class="secondary" onclick="prevImg()">Previous</button>
            <button onclick="rotateImg('ccw')">Rotate 90° CCW</button>
            <button onclick="rotateImg('cw')">Rotate 90° CW</button>
            <button onclick="nextImg()">Next</button>
            <button class="secondary" onclick="backToSetup()">Back</button>
        </div>
    </div>

    <script>
        let outOfRangeImages = [];
        let currentIndex = 0;

        async function runAnalysis() {
            const btn = event.target;
            btn.innerText = "Analyzing...";
            const res = await fetch('/api/analyze');
            const data = await res.json();
            document.getElementById('histogram').src = "data:image/png;base64," + data.histogram;
            document.getElementById('analysis-output').classList.remove('hidden');
            btn.innerText = "Re-run Analysis";
        }

        async function startReview() {
            const min = document.getElementById('min-ratio').value;
            const max = document.getElementById('max-ratio').value;
            const res = await fetch(`/api/filter?min=${min}&max=${max}`);
            outOfRangeImages = await res.json();
            
            if (outOfRangeImages.length === 0) {
                alert("No images out of range!");
                return;
            }
            
            currentIndex = 0;
            document.getElementById('setup-step').classList.add('hidden');
            document.getElementById('review-step').classList.remove('hidden');
            showCurrentImage();
        }

        function showCurrentImage() {
            if (currentIndex < 0 || currentIndex >= outOfRangeImages.length) return;
            const item = outOfRangeImages[currentIndex];
            document.getElementById('current-index-display').innerText = currentIndex + 1;
            document.getElementById('total-count-display').innerText = outOfRangeImages.length;
            document.getElementById('img-info').innerText = `${item.filename} (Ratio: ${item.ratio.toFixed(2)})`;
            // Add cache buster
            document.getElementById('review-img').src = `/images/${item.filename}?t=${new Date().getTime()}`;
        }

        async function rotateImg(direction) {
            const item = outOfRangeImages[currentIndex];
            const res = await fetch('/api/rotate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: item.filename, direction: direction })
            });
            const data = await res.json();
            if (data.success) {
                showCurrentImage();
            } else {
                alert("Rotate failed: " + data.error);
            }
        }

        function nextImg() {
            if (currentIndex < outOfRangeImages.length - 1) {
                currentIndex++;
                showCurrentImage();
            } else {
                alert("Reached the end!");
            }
        }

        function prevImg() {
            if (currentIndex > 0) {
                currentIndex--;
                showCurrentImage();
            }
        }

        function backToSetup() {
            document.getElementById('setup-step').classList.remove('hidden');
            document.getElementById('review-step').classList.add('hidden');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze')
def api_analyze():
    results = analyze_images()
    hist_base64 = generate_histogram_base64(results)
    return jsonify({'count': len(results), 'histogram': hist_base64})

@app.route('/api/filter')
def api_filter():
    min_ratio = float(request.args.get('min', 0))
    max_ratio = float(request.args.get('max', 999))
    filtered = [d for d in analysis_results if d['ratio'] < min_ratio or d['ratio'] > max_ratio]
    return jsonify(filtered)

@app.route('/api/rotate', methods=['POST'])
def api_rotate():
    data = request.json
    filename = data.get('filename')
    direction = data.get('direction') # 'cw' or 'ccw'
    
    path = os.path.join(IMAGE_DIR, filename)
    try:
        with Image.open(path) as img:
            angle = -90 if direction == 'cw' else 90
            rotated = img.rotate(angle, expand=True)
            rotated.save(path)
        
        # Update ratio in current results
        for item in analysis_results:
            if item['filename'] == filename:
                w, h = rotated.size
                item['width'], item['height'] = w, h
                item['ratio'] = w / h
                break
                
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    print("Starting Web UI on http://0.0.0.0:5000")
    print("If running in a remote server, please use SSH port forwarding:")
    print("ssh -L 5000:localhost:5000 your-server-ip")
    app.run(host='0.0.0.0', port=5000, debug=False)
