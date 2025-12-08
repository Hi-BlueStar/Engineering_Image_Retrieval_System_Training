import os
import sys
import time

# 引入之前的搜尋邏輯庫 (假設您已安裝相關套件)
import chromadb
import numpy as np
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from PIL import Image
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


# ==========================================
# 1. 全域樣式設定 (The Sci-Fi Skin)
# ==========================================
# 色票: 深空黑背景, 霓虹青(Cyan)為主色, 警示橘(Amber)為強調色
THEME_COLOR_BG = "#050a14"
THEME_COLOR_PANEL = "rgba(16, 32, 55, 0.85)"
THEME_COLOR_ACCENT = "#00f3ff"  # Cyan Neon
THEME_COLOR_TEXT = "#a6e2ff"
THEME_COLOR_WARN = "#ffae00"
THEME_FONT = "Segoe UI"  # 或 Orbitron 如果有安裝

STYLESHEET = f"""
QMainWindow {{
    background-color: {THEME_COLOR_BG};
}}

/* 通用 Frame 樣式 - 類似飛船面板 */
QFrame#MainPanel, QFrame#SidePanel, QFrame#ResultPanel {{
    background-color: {THEME_COLOR_PANEL};
    border: 1px solid #1c3a5e;
    border-radius: 8px;
}}

QFrame#ResultCard {{
    background-color: rgba(0, 20, 40, 0.6);
    border: 1px solid #1c3a5e;
    border-radius: 0px;
}}
QFrame#ResultCard:hover {{
    border: 1px solid {THEME_COLOR_ACCENT};
    background-color: rgba(0, 243, 255, 0.1);
}}

/* 標籤文字 */
QLabel {{
    color: {THEME_COLOR_TEXT};
    font-family: '{THEME_FONT}';
    font-size: 14px;
}}
QLabel#TitleLabel {{
    font-size: 24px;
    font-weight: bold;
    color: {THEME_COLOR_ACCENT};
    letter-spacing: 2px;
}}
QLabel#StatusLabel {{
    color: {THEME_COLOR_WARN};
    font-family: 'Consolas', monospace;
    font-size: 12px;
}}

/* 輸入框 - 類似終端機輸入 */
QLineEdit {{
    background-color: rgba(0, 0, 0, 0.5);
    border: 1px solid #1c3a5e;
    border-bottom: 2px solid {THEME_COLOR_ACCENT};
    color: #fff;
    padding: 8px;
    font-family: 'Consolas', monospace;
    font-size: 14px;
}}
QLineEdit:focus {{
    border: 1px solid {THEME_COLOR_ACCENT};
    background-color: rgba(0, 243, 255, 0.05);
}}

/* 按鈕 - 科幻風格 */
QPushButton {{
    background-color: rgba(0, 243, 255, 0.1);
    border: 1px solid {THEME_COLOR_ACCENT};
    color: {THEME_COLOR_ACCENT};
    padding: 10px 20px;
    font-weight: bold;
    letter-spacing: 1px;
    text-transform: uppercase;
}}
QPushButton:hover {{
    background-color: {THEME_COLOR_ACCENT};
    color: #000;
    box-shadow: 0 0 10px {THEME_COLOR_ACCENT};
}}
QPushButton:pressed {{
    background-color: #008c9e;
    border: 1px solid #fff;
}}
QPushButton:disabled {{
    border-color: #555;
    color: #555;
    background-color: transparent;
}}

/* 捲軸美化 */
QScrollBar:vertical {{
    border: none;
    background: #0b1521;
    width: 10px;
    margin: 0px 0px 0px 0px;
}}
QScrollBar::handle:vertical {{
    background: {THEME_COLOR_ACCENT};
    min-height: 20px;
    border-radius: 2px;
}}
"""


# ==========================================
# 2. 後端邏輯 (Worker Thread)
# ==========================================
class SearchWorker(QThread):
    """
    在背景執行緒處理 ChromaDB 的載入與搜尋，避免凍結 GUI。
    """

    results_ready = Signal(list)
    status_update = Signal(str)

    def __init__(self):
        super().__init__()
        self.collection = None
        self.mode = None  # 'text' or 'image'
        self.query = None
        self.db_path = "./chroma_db_store"  # 請確認此路徑正確
        self.collection_name = "engineering_components_v1"

    def run(self):
        # 1. 初始化 (如果是第一次)
        if not self.collection:
            self.status_update.emit("SYSTEM_INIT: 連接神經網路資料庫...")
            try:
                client = chromadb.PersistentClient(path=self.db_path)
                embedding_func = OpenCLIPEmbeddingFunction()
                self.collection = client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_func,
                    data_loader=ImageLoader(),
                )
                self.status_update.emit("SYSTEM_READY: 資料庫連線成功。")
            except Exception as e:
                self.status_update.emit(f"SYSTEM_ERROR: 資料庫連接失敗 - {str(e)}")
                return

        # 2. 執行搜尋
        if not self.query:
            return

        try:
            results = None
            if self.mode == "text":
                self.status_update.emit(
                    f"SEARCH_PROTOCOL: 正在解析語意向量 '{self.query}'..."
                )
                results = self.collection.query(
                    query_texts=[self.query],
                    n_results=10,  # 顯示前 10 筆
                    include=["metadatas", "distances", "uris"],
                )
            elif self.mode == "image":
                self.status_update.emit("VISUAL_SCAN: 正在比對特徵向量...")
                img_array = np.array(Image.open(self.query))
                results = self.collection.query(
                    query_images=[img_array],
                    n_results=10,
                    include=["metadatas", "distances", "uris"],
                )

            # 3. 整理結果並回傳
            formatted_results = []
            if results and results["ids"]:
                ids = results["ids"][0]
                dists = results["distances"][0]
                metas = results["metadatas"][0]
                uris = results["uris"][0]

                for i in range(len(ids)):
                    formatted_results.append(
                        {
                            "id": ids[i],
                            "score": 1 - dists[i],  # 簡單轉為相似度
                            "meta": metas[i],
                            "uri": uris[i],
                        }
                    )

            self.results_ready.emit(formatted_results)
            self.status_update.emit("OP_COMPLETE: 檢索完成。")

        except Exception as e:
            self.status_update.emit(f"CRITICAL_ERROR: {str(e)}")

    def search_text(self, text):
        self.mode = "text"
        self.query = text
        self.start()

    def search_image(self, image_path):
        self.mode = "image"
        self.query = image_path
        self.start()


# ==========================================
# 3. GUI 元件
# ==========================================


class ResultCard(QFrame):
    """
    單個搜尋結果的展示卡片
    """

    def __init__(self, data):
        super().__init__()
        self.setObjectName("ResultCard")
        self.setFixedSize(220, 280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 圖片區域
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background-color: #000; border: 1px dashed #333;")
        self.img_label.setFixedSize(208, 150)

        # 嘗試載入圖片
        pixmap = QPixmap(data["uri"])
        if not pixmap.isNull():
            self.img_label.setPixmap(
                pixmap.scaled(200, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.img_label.setText("IMG_ERR")

        # 資訊區域
        info_layout = QVBoxLayout()

        # 相似度條
        score = data["score"]
        score_label = QLabel(f"MATCH: {score:.1%}")
        score_label.setStyleSheet(f"color: {THEME_COLOR_ACCENT}; font-weight: bold;")

        # 檔名與料號
        fname = data["meta"].get("filename", "N/A")
        part_id = data["meta"].get("part_id", "Unknown")

        name_label = QLabel(fname)
        name_label.setWordWrap(True)
        name_label.setStyleSheet("font-size: 11px; color: #fff;")

        id_label = QLabel(f"ID: {part_id}")
        id_label.setStyleSheet("font-size: 10px; color: #888;")

        layout.addWidget(self.img_label)
        layout.addWidget(score_label)
        layout.addWidget(name_label)
        layout.addWidget(id_label)
        layout.addStretch()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ATC 智慧檢索系統 // NEURAL SEARCH LINK")
        self.resize(1200, 800)

        # 初始化後端 Worker
        self.worker = SearchWorker()
        self.worker.status_update.connect(self.update_log)
        self.worker.results_ready.connect(self.display_results)

        # 設定主介面
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(15, 15, 15, 15)

        self._setup_ui()
        self._apply_animations()

        # 啟動時自動初始化 DB
        self.worker.start()

    def _setup_ui(self):
        # --- 左側控制面板 (Side Panel) ---
        side_panel = QFrame()
        side_panel.setObjectName("SidePanel")
        side_panel.setFixedWidth(320)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setSpacing(20)

        # 1. 標題與 Logo 區
        title_label = QLabel("NEURAL\nENGINEERING\nARCHIVE")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignLeft)

        subtitle_label = QLabel("v4.0.1 [CONNECTED]")
        subtitle_label.setStyleSheet(
            f"color: {THEME_COLOR_ACCENT}; letter-spacing: 1px;"
        )

        # 2. 搜尋模式切換
        self.mode_label = QLabel(":: SELECT SEARCH PROTOCOL ::")
        self.mode_label.setStyleSheet("font-weight: bold; margin-top: 20px;")

        # 3. 文字搜尋區塊
        input_container = QFrame()
        input_container.setStyleSheet("background: transparent; border: none;")
        input_vbox = QVBoxLayout(input_container)
        input_vbox.setContentsMargins(0, 0, 0, 0)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("輸入特徵關鍵字 (例: 大型鈑金)...")
        self.text_input.returnPressed.connect(self.on_text_search)

        btn_text_search = QPushButton("INITIATE TEXT SCAN")
        btn_text_search.clicked.connect(self.on_text_search)

        input_vbox.addWidget(QLabel("TEXT INPUT:"))
        input_vbox.addWidget(self.text_input)
        input_vbox.addWidget(btn_text_search)

        # 4. 圖片搜尋區塊
        img_container = QFrame()
        img_container.setStyleSheet("background: transparent; border: none;")
        img_vbox = QVBoxLayout(img_container)
        img_vbox.setContentsMargins(0, 0, 0, 0)

        self.img_path_label = QLabel("NO IMAGE LOADED")
        self.img_path_label.setStyleSheet(
            "color: #666; font-style: italic; border: 1px dashed #444; padding: 10px;"
        )
        self.img_path_label.setAlignment(Qt.AlignCenter)

        btn_load_img = QPushButton("LOAD SOURCE IMAGE")
        btn_load_img.clicked.connect(self.browse_image)

        self.btn_img_search = QPushButton("ANALYZE VISUAL DATA")
        self.btn_img_search.setEnabled(False)
        self.btn_img_search.clicked.connect(self.on_image_search)

        img_vbox.addWidget(QLabel("VISUAL INPUT:"))
        img_vbox.addWidget(self.img_path_label)
        img_vbox.addWidget(btn_load_img)
        img_vbox.addWidget(self.btn_img_search)

        # 分隔線
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {THEME_COLOR_ACCENT};")

        # 5. 系統日誌 (Log)
        log_label = QLabel(":: SYSTEM LOG ::")
        self.log_area = QLabel("System Initializing...\nWaiting for core...")
        self.log_area.setObjectName("StatusLabel")
        self.log_area.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.log_area.setWordWrap(True)
        self.log_area.setStyleSheet(
            "background: rgba(0,0,0,0.3); padding: 5px; border: none;"
        )
        self.log_area.setFixedHeight(150)

        # 組裝左側
        side_layout.addWidget(title_label)
        side_layout.addWidget(subtitle_label)
        side_layout.addWidget(self.mode_label)
        side_layout.addWidget(input_container)
        side_layout.addWidget(line)
        side_layout.addWidget(img_container)
        side_layout.addStretch()
        side_layout.addWidget(log_label)
        side_layout.addWidget(self.log_area)

        # --- 右側結果面板 (Main Panel) ---
        main_panel = QFrame()
        main_panel.setObjectName("MainPanel")
        main_layout_v = QVBoxLayout(main_panel)

        # 頂部狀態條
        header_layout = QHBoxLayout()
        self.result_status = QLabel("STATUS: IDLE // WAITING FOR INPUT")
        self.result_status.setStyleSheet("font-size: 16px; color: #fff;")

        # 裝飾性元素
        deco_box = QLabel("[ SECURE CONNECTION ]")
        deco_box.setStyleSheet(
            f"border: 1px solid {THEME_COLOR_ACCENT}; color: {THEME_COLOR_ACCENT}; padding: 2px 8px;"
        )

        header_layout.addWidget(self.result_status)
        header_layout.addStretch()
        header_layout.addWidget(deco_box)

        # 結果顯示區 (Scroll Area + Grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background: transparent;")
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(15)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        scroll.setWidget(self.scroll_content)

        main_layout_v.addLayout(header_layout)
        main_layout_v.addWidget(scroll)

        # 加入佈局
        self.main_layout.addWidget(side_panel)
        self.main_layout.addWidget(main_panel, stretch=1)

    def _apply_animations(self):
        # 這裡可以加入一些簡單的啟動動畫，例如透明度漸變
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

        self.anim = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.anim.setDuration(1500)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)
        self.anim.start()

    # --- Slot Functions ---

    def update_log(self, message):
        current_text = self.log_area.text()
        # 保持日誌在最後 10 行
        lines = current_text.split("\n")
        if len(lines) > 8:
            lines = lines[-8:]

        timestamp = time.strftime("%H:%M:%S")
        new_line = f"[{timestamp}] {message}"
        self.log_area.setText("\n".join(lines + [new_line]))

        # 更新狀態條
        if "SEARCH" in message or "SCAN" in message:
            self.result_status.setText(f"STATUS: PROCESSING... >> {message}")
            self.result_status.setStyleSheet(
                f"color: {THEME_COLOR_WARN}; font-size: 16px;"
            )
        elif "COMPLETE" in message:
            self.result_status.setText("STATUS: RESULTS ACQUIRED")
            self.result_status.setStyleSheet(
                f"color: {THEME_COLOR_ACCENT}; font-size: 16px;"
            )

    def on_text_search(self):
        text = self.text_input.text().strip()
        if not text:
            self.update_log("WARNING: Input empty.")
            return

        self.clear_results()
        self.worker.search_text(text)

    def browse_image(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Select Source Image", ".", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if fname:
            self.current_img_path = fname
            self.img_path_label.setText(os.path.basename(fname))
            # 顯示預覽縮圖
            # 1. 先在外部處理路徑的反斜線 (Windows 路徑問題)
            formatted_path = fname.replace("\\", "/")

            # 2. 將處理好的變數放入 f-string
            self.img_path_label.setStyleSheet(f"""
                border: 2px solid {THEME_COLOR_ACCENT}; 
                background-image: url({formatted_path}); 
                background-position: center; 
                background-repeat: no-repeat;
                color: transparent;
            """)

            self.btn_img_search.setEnabled(True)
            self.update_log(f"IMG_LOAD: {os.path.basename(fname)}")

    def on_image_search(self):
        if hasattr(self, "current_img_path"):
            self.clear_results()
            self.worker.search_image(self.current_img_path)

    def clear_results(self):
        # 清除 Grid 中的所有 widget
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def display_results(self, results):
        self.clear_results()

        if not results:
            no_res = QLabel("NO MATCHING DATA FOUND IN ARCHIVE.")
            no_res.setStyleSheet("font-size: 20px; color: #666; margin-top: 50px;")
            self.grid_layout.addWidget(no_res, 0, 0)
            return

        row = 0
        col = 0
        max_cols = 4  # 每行顯示幾個

        for data in results:
            card = ResultCard(data)
            self.grid_layout.addWidget(card, row, col)

            col += 1
            if col >= max_cols:
                col = 0
                row += 1


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 套用全域樣式
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
