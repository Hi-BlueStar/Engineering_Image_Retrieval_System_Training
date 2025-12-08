import sys
import os
import shutil
import uuid
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                               QScrollArea, QGridLayout, QLineEdit, QProgressBar,
                               QFrame, QMessageBox, QSizePolicy, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QUrl
from PySide6.QtGui import QPixmap, QIcon, QColor, QDragEnterEvent, QDropEvent, QImage

# ==========================================
# 後端邏輯 (Backend Logic)
# ==========================================

class ChromaBackend:
    def __init__(self, persist_path="./chroma_data"):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_func = OpenCLIPEmbeddingFunction()
        self.data_loader = ImageLoader()
        self.collection = self.client.get_or_create_collection(
            name="engineering_components_gallery",
            embedding_function=self.embedding_func,
            data_loader=self.data_loader
        )

    def search(self, query_image_path, top_k=20):
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"找不到圖片: {query_image_path}")

        results = self.collection.query(
            query_uris=[query_image_path],
            n_results=top_k,
            include=['metadatas', 'distances', 'uris'] 
        )

        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        uris = results['uris'][0] if 'uris' in results else []

        if not metadatas:
            return pd.DataFrame()

        parsed_data = []
        for i, (meta, dist) in enumerate(zip(metadatas, distances)):
            # 處理可能缺失的 URI
            img_uri = uris[i] if i < len(uris) else ""
            
            similarity = 1 - dist
            parsed_data.append({
                "item_name": meta.get("item_name", "Unknown"),
                "category": meta.get("category", "General"),
                "similarity": similarity,
                "filename": meta.get("filename", os.path.basename(img_uri)),
                "filepath": img_uri
            })
        
        return pd.DataFrame(parsed_data)

    def add_images(self, file_paths):
        """新增圖片到資料庫，自動提取簡單的 metadata"""
        ids = []
        uris = []
        metadatas = []

        for path in file_paths:
            if not os.path.exists(path):
                continue
                
            # 簡單的 metadata 推斷策略
            filename = os.path.basename(path)
            item_name = os.path.splitext(filename)[0]
            # 假設上一層資料夾名稱為分類
            category = os.path.basename(os.path.dirname(path)) 
            
            unique_id = str(uuid.uuid4())
            ids.append(unique_id)
            uris.append(path)
            metadatas.append({
                "filename": filename,
                "item_name": item_name,
                "category": category,
                "type": "user_added"
            })

        if ids:
            self.collection.add(
                ids=ids,
                uris=uris,
                metadatas=metadatas
            )
        return len(ids)

# ==========================================
# 多執行緒工作 (Workers)
# ==========================================

class SearchWorker(QThread):
    finished = Signal(object) # 回傳 DataFrame
    error = Signal(str)

    def __init__(self, backend, image_path, top_k):
        super().__init__()
        self.backend = backend
        self.image_path = image_path
        self.top_k = top_k

    def run(self):
        try:
            df = self.backend.search(self.image_path, self.top_k)
            self.finished.emit(df)
        except Exception as e:
            self.error.emit(str(e))

class AddImageWorker(QThread):
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, backend, file_paths):
        super().__init__()
        self.backend = backend
        self.file_paths = file_paths

    def run(self):
        try:
            count = self.backend.add_images(self.file_paths)
            self.finished.emit(count)
        except Exception as e:
            self.error.emit(str(e))

# ==========================================
# UI 元件與樣式 (Styles & Widgets)
# ==========================================

# 新擬態暗色主題樣式表
STYLESHEET = """
QMainWindow {
    background-color: #2b2e33;
}
QWidget {
    font-family: "Microsoft JhengHei", "Segoe UI", sans-serif;
    color: #e0e0e0;
    font-size: 14px;
}
/* 新擬態卡片容器 */
QFrame#NeumorphPanel {
    background-color: #2b2e33;
    border-radius: 15px;
    border: 1px solid #36393f;
}
/* 按鈕樣式 */
QPushButton {
    background-color: #2b2e33;
    border-radius: 10px;
    padding: 10px;
    color: #aaaaaa;
    border: 2px solid #23262a;
    font-weight: bold;
}
QPushButton:hover {
    color: #ffffff;
    background-color: #32353a;
    border: 2px solid #3a3e44;
}
QPushButton:pressed {
    background-color: #23262a;
    border: 2px solid #1f2125;
}
/* 搜尋按鈕特別樣式 */
QPushButton#PrimaryButton {
    background-color: #2b2e33;
    color: #4db8ff;
    border: 2px solid #2b2e33;
}
QPushButton#PrimaryButton:hover {
    color: #70c9ff;
    border: 2px solid #3a3e44;
}
/* 滾動條 */
QScrollBar:vertical {
    border: none;
    background: #2b2e33;
    width: 8px;
    margin: 0px 0px 0px 0px;
}
QScrollBar::handle:vertical {
    background: #40444b;
    min-height: 20px;
    border-radius: 4px;
}
"""

class ImageDropZone(QLabel):
    """支援拖放圖片的顯示區域"""
    imageDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("拖曳圖片至此\n或點擊選取")
        self.setStyleSheet("""
            QLabel {
                background-color: #272a2e;
                border: 2px dashed #40444b;
                border-radius: 15px;
                color: #777;
                font-weight: bold;
            }
            QLabel:hover {
                border-color: #4db8ff;
                color: #4db8ff;
            }
        """)
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 300)
        self.current_path = None

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(self, "選取查詢圖片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.load_image(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.load_image(file_path)

    def load_image(self, path):
        self.current_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            self.imageDropped.emit(path)

class ResultCard(QFrame):
    """顯示單張搜尋結果的卡片"""
    def __init__(self, data):
        super().__init__()
        self.setObjectName("NeumorphPanel")
        self.setFixedSize(220, 260)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 圖片
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border: none; background-color: transparent;")
        
        # 載入圖片 (處理路徑)
        img_path = data.get('filepath', '')
        if os.path.exists(img_path):
            pix = QPixmap(img_path)
            self.img_label.setPixmap(pix.scaled(180, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.img_label.setText("圖片遺失")

        # 文字資訊
        info_layout = QVBoxLayout()
        name_label = QLabel(data.get('item_name', 'Unknown'))
        name_label.setStyleSheet("font-weight: bold; color: #fff; font-size: 13px;")
        name_label.setWordWrap(True)
        
        score = data.get('similarity', 0.0)
        score_color = "#4db8ff" if score > 0.8 else "#aaaaaa"
        score_label = QLabel(f"相似度: {score:.4f}")
        score_label.setStyleSheet(f"color: {score_color}; font-size: 12px;")

        filename_label = QLabel(data.get('filename', ''))
        filename_label.setStyleSheet("color: #666; font-size: 10px;")
        filename_label.setWordWrap(True)

        info_layout.addWidget(name_label)
        info_layout.addWidget(score_label)
        info_layout.addWidget(filename_label)
        info_layout.addStretch()

        layout.addWidget(self.img_label)
        layout.addLayout(info_layout)

        # 添加陰影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(3)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

# ==========================================
# 主視窗 (Main Window)
# ==========================================

class EngineeringSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("工程元件智慧檢索系統 - AI 視覺辨識")
        self.resize(1200, 800)
        self.setStyleSheet(STYLESHEET)
        
        # 初始化後端
        self.backend = ChromaBackend()
        self.current_k = 10  # 初始搜尋數量
        self.current_query_path = None
        self.loading = False

        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setSpacing(20)

        # --- 左側面板 (控制與查詢區) ---
        left_panel = QFrame()
        left_panel.setObjectName("NeumorphPanel")
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        # 標題
        title_label = QLabel("查詢配置")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #fff;")
        left_layout.addWidget(title_label)

        # 圖片拖放區
        self.drop_zone = ImageDropZone()
        self.drop_zone.imageDropped.connect(self.on_query_image_selected)
        left_layout.addWidget(self.drop_zone)
        
        # 圖片名稱顯示
        self.lbl_filename = QLabel("尚未選擇圖片")
        self.lbl_filename.setStyleSheet("color: #888; margin-top: 5px;")
        self.lbl_filename.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_filename)

        # 分隔線
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #444;")
        left_layout.addWidget(line)

        # 資料庫管理區
        db_title = QLabel("資料庫管理")
        db_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fff; margin-top: 10px;")
        left_layout.addWidget(db_title)

        btn_add_files = QPushButton("新增圖片 (選取檔案)")
        btn_add_files.clicked.connect(self.add_files)
        
        btn_add_folder = QPushButton("批次匯入 (選取資料夾)")
        btn_add_folder.clicked.connect(self.add_folder)

        # 啟用拖放至按鈕區域 (選做，這裡用按鈕為主)
        self.setAcceptDrops(True) 

        left_layout.addWidget(btn_add_files)
        left_layout.addWidget(btn_add_folder)
        left_layout.addStretch()
        
        # 搜尋按鈕
        self.btn_search = QPushButton("開始檢索")
        self.btn_search.setObjectName("PrimaryButton")
        self.btn_search.setMinimumHeight(50)
        self.btn_search.setStyleSheet("font-size: 16px;")
        self.btn_search.clicked.connect(lambda: self.start_search(reset=True))
        left_layout.addWidget(self.btn_search)

        # --- 右側面板 (結果顯示區) ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 頂部狀態列
        header_layout = QHBoxLayout()
        self.lbl_status = QLabel("準備就緒")
        self.lbl_status.setStyleSheet("font-size: 20px; font-weight: bold; color: #4db8ff;")
        header_layout.addWidget(self.lbl_status)
        header_layout.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Infinite loop style
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: none; background: #333; height: 4px; border-radius: 2px;}
            QProgressBar::chunk { background: #4db8ff; border-radius: 2px;}
        """)
        header_layout.addWidget(self.progress_bar)
        right_layout.addLayout(header_layout)

        # 捲動區域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")
        
        self.results_container = QWidget()
        self.results_grid = QGridLayout(self.results_container)
        self.results_grid.setSpacing(20)
        self.scroll_area.setWidget(self.results_container)
        
        right_layout.addWidget(self.scroll_area)

        # 底部 "搜尋更多"
        self.btn_load_more = QPushButton("搜尋更多結果 (+10)")
        self.btn_load_more.setVisible(False)
        self.btn_load_more.clicked.connect(self.load_more)
        right_layout.addWidget(self.btn_load_more)

        # 添加到主佈局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1)

    # --- 邏輯功能 ---

    def on_query_image_selected(self, path):
        self.current_query_path = path
        self.lbl_filename.setText(os.path.basename(path))
        self.lbl_status.setText("圖片已載入，請點擊檢索")
        self.btn_load_more.setVisible(False)

    def start_search(self, reset=True):
        if not self.current_query_path:
            QMessageBox.warning(self, "警告", "請先拖入或選擇一張查詢圖片")
            return
        
        if reset:
            self.current_k = 10
            # 清空現有結果
            for i in reversed(range(self.results_grid.count())): 
                self.results_grid.itemAt(i).widget().setParent(None)
        
        self.set_loading(True, "正在 AI 資料庫中檢索...")
        
        # 啟動搜尋執行緒
        self.search_worker = SearchWorker(self.backend, self.current_query_path, self.current_k)
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.error.connect(self.on_error)
        self.search_worker.start()

    def load_more(self):
        self.current_k += 10
        self.start_search(reset=True) # 簡單起見，重新搜尋更大的 K 值 (Chroma 查詢特性)

    def on_search_finished(self, df):
        self.set_loading(False, f"檢索完成，共找到 {len(df)} 筆相關結果")
        
        # 為了避免畫面閃爍，這裡雖然是 reset=True 時全部重繪，
        # 但如果是 pagination 邏輯可以優化。
        # 這裡簡單處理：全部重畫
        for i in reversed(range(self.results_grid.count())): 
            self.results_grid.itemAt(i).widget().setParent(None)

        if df.empty:
            self.lbl_status.setText("未找到相似結果")
            return

        row, col = 0, 0
        max_cols = 4 # 每行顯示幾個
        
        for index, row_data in df.iterrows():
            card = ResultCard(row_data)
            self.results_grid.addWidget(card, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        self.btn_load_more.setVisible(True)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg)")
        if files:
            self.start_add_worker(files)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder:
            images = []
            for root, dirs, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        images.append(os.path.join(root, f))
            if images:
                self.start_add_worker(images)
            else:
                QMessageBox.information(self, "提示", "資料夾內沒有圖片")

    def start_add_worker(self, file_paths):
        self.set_loading(True, f"正在將 {len(file_paths)} 張圖片向量化並存入資料庫...")
        self.add_worker = AddImageWorker(self.backend, file_paths)
        self.add_worker.finished.connect(self.on_add_finished)
        self.add_worker.error.connect(self.on_error)
        self.add_worker.start()

    def on_add_finished(self, count):
        self.set_loading(False, f"成功新增 {count} 張圖片至資料庫")
        QMessageBox.information(self, "成功", f"已新增 {count} 張圖片。\n您可以立即對這些圖片進行搜尋。")

    def set_loading(self, is_loading, message):
        self.loading = is_loading
        self.lbl_status.setText(message)
        self.progress_bar.setVisible(is_loading)
        self.btn_search.setEnabled(not is_loading)
        if is_loading:
            self.setCursor(Qt.WaitCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def on_error(self, err_msg):
        self.set_loading(False, "發生錯誤")
        QMessageBox.critical(self, "錯誤", err_msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 設定全域字型
    font = app.font()
    font.setFamily("Microsoft JhengHei")
    app.setFont(font)
    
    window = EngineeringSearchApp()
    window.show()
    sys.exit(app.exec())