import sys
import time
import uuid
from pathlib import Path

# ChromaDB Imports
import chromadb
import numpy as np
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from PIL import Image
from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    QThread,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)

# PySide6 Imports
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


# ==========================================
# 0. EVA 風格定義 (MAGI System Style)
# ==========================================
class NERV:
    COLOR_BG = "#0D0D0D"  # 深黑背景
    COLOR_MAIN = "#FF9900"  # MAGI 橘色 (主要)
    COLOR_ACCENT = "#39FF14"  # 螢光綠 (同步率高)
    COLOR_DANGER = "#EC0000"  # 警示紅 (使徒/錯誤)
    COLOR_DIM = "rgba(255, 153, 0, 0.3)"  # 半透明橘色
    FONT_MAIN = "Microsoft JhengHei UI"  # 繁中字體
    FONT_TECH = "Consolas"  # 數據顯示字體
    FONT_HEADER = "Impact"  # 標題字體


STYLE_SHEET = f"""
QMainWindow {{
    background-color: {NERV.COLOR_BG};
    background-image: url('none'); /* 可自行加入蜂巢背景圖 */
}}

/* Tab Widget - 類似機密檔案夾 */
QTabWidget::pane {{
    border: 2px solid {NERV.COLOR_MAIN};
    background: rgba(13, 13, 13, 0.9);
}}
QTabBar::tab {{
    background: #000;
    color: {NERV.COLOR_MAIN};
    border: 1px solid {NERV.COLOR_MAIN};
    padding: 8px 20px;
    margin-right: 2px;
    font-family: '{NERV.FONT_HEADER}';
    font-size: 14px;
    letter-spacing: 2px;
}}
QTabBar::tab:selected {{
    background: {NERV.COLOR_MAIN};
    color: #000;
    font-weight: bold;
}}

/* ScrollArea */
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: #111;
    width: 12px;
}}
QScrollBar::handle:vertical {{
    background: {NERV.COLOR_MAIN};
    min-height: 20px;
}}

/* Inputs */
QLineEdit {{
    background-color: rgba(0, 0, 0, 0.8);
    border: 1px solid {NERV.COLOR_MAIN};
    color: {NERV.COLOR_MAIN};
    font-family: '{NERV.FONT_TECH}';
    font-size: 16px;
    padding: 5px;
    selection-background-color: {NERV.COLOR_MAIN};
    selection-color: #000;
}}
QLineEdit:focus {{
    border: 2px solid {NERV.COLOR_ACCENT};
    color: {NERV.COLOR_ACCENT};
}}

/* Log Area */
QLabel#LogLabel {{
    color: {NERV.COLOR_MAIN};
    font-family: '{NERV.FONT_TECH}';
    font-size: 12px;
}}
"""

# ==========================================
# 1. 自定義 EVA 元件 (Vector Drawing)
# ==========================================


class EvaFrame(QFrame):
    """
    帶有科幻角落切角與十字準心的 Frame
    """

    def __init__(self, color=NERV.COLOR_MAIN, parent=None):
        super().__init__(parent)
        self.color_hex = color
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(self.color_hex))
        pen.setWidth(2)
        painter.setPen(pen)

        rect = self.rect()
        w, h = rect.width(), rect.height()
        cut = 15  # 切角大小

        # 繪製切角矩形路徑
        path = QPainterPath()
        path.moveTo(cut, 0)
        path.lineTo(w - cut, 0)
        path.lineTo(w, cut)
        path.lineTo(w, h - cut)
        path.lineTo(w - cut, h)
        path.lineTo(cut, h)
        path.lineTo(0, h - cut)
        path.lineTo(0, cut)
        path.closeSubpath()

        painter.drawPath(path)

        # 繪製裝飾性細節 (Tech Lines)
        pen.setWidth(1)
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        # 繪製內部十字線
        painter.drawLine(w // 2, 10, w // 2, h - 10)
        painter.drawLine(10, h // 2, w - 10, h // 2)


class EvaButton(QPushButton):
    """
    EVA 風格按鈕：滑鼠移入會有掃描動畫與顏色變化
    """

    def __init__(self, text, color=NERV.COLOR_MAIN, parent=None):
        super().__init__(text, parent)
        self.base_color = color
        self.hover_progress = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setMinimumHeight(40)
        self.setFont(QFont(NERV.FONT_HEADER, 12))

        # 動畫設定
        self.anim = QPropertyAnimation(self, b"hover_progress_prop")
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

    def set_hover_progress(self, val):
        self.hover_progress = val
        self.update()

    def get_hover_progress(self):
        return self.hover_progress

    hover_progress_prop = Property(float, get_hover_progress, set_hover_progress)

    def enterEvent(self, event):
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.start()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        w, h = rect.width(), rect.height()

        # 計算當前顏色
        is_hover = self.hover_progress > 0.01
        current_color = (
            QColor(NERV.COLOR_ACCENT) if is_hover else QColor(self.base_color)
        )
        bg_alpha = int(50 + (150 * self.hover_progress))

        # 背景
        bg_color = QColor(current_color)
        bg_color.setAlpha(bg_alpha)
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)

        # 六邊形或切角形狀
        path = QPainterPath()
        cut = 10
        path.moveTo(cut, 0)
        path.lineTo(w, 0)
        path.lineTo(w, h - cut)
        path.lineTo(w - cut, h)
        path.lineTo(0, h)
        path.lineTo(0, cut)
        path.closeSubpath()

        painter.drawPath(path)

        # 邊框
        pen = QPen(current_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        # 掃描線效果 (Scanline)
        if self.hover_progress > 0:
            scan_y = int(h * self.hover_progress)
            pen.setWidth(1)
            pen.setColor(QColor(255, 255, 255, 180))
            painter.setPen(pen)
            painter.drawLine(0, scan_y, w, scan_y)

        # 文字
        painter.setPen(QColor("#000") if is_hover else current_color)
        painter.setFont(self.font())
        painter.drawText(rect, Qt.AlignCenter, self.text())


class DropZone(QLabel):
    """
    支援拖拉的圖片放置區，帶有閃爍動畫
    """

    fileDropped = Signal(str)

    def __init__(self, text="DRAG & DROP IMAGE HERE", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setFont(QFont(NERV.FONT_HEADER, 14))
        self.default_text = text
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {NERV.COLOR_MAIN};
                color: {NERV.COLOR_MAIN};
                background: rgba(255, 153, 0, 0.05);
            }}
        """)

        # 呼吸燈動畫
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_border)
        self.pulse = 0
        self.timer.start(100)

    def animate_border(self):
        self.pulse = (self.pulse + 1) % 20
        alpha = 0.05 + (0.1 * abs(10 - self.pulse) / 10)
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px dashed {NERV.COLOR_MAIN};
                color: {NERV.COLOR_MAIN};
                background: rgba(255, 153, 0, {alpha:.2f});
            }}
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.setText(":: LINK ESTABLISHED ::")
            self.setStyleSheet(
                f"border: 2px solid {NERV.COLOR_ACCENT}; color: {NERV.COLOR_ACCENT};"
            )
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.fileDropped.emit(file_path)
            self.setText(":: DATA LOADED ::")


# ==========================================
# 2. 後端 Worker (多執行緒)
# ==========================================


class DatabaseWorker(QThread):
    status_signal = Signal(str, str)  # level, message
    search_result_signal = Signal(list)
    import_progress_signal = Signal(int, int)  # current, total

    def __init__(self):
        super().__init__()
        self.task = None  # 'init', 'search_text', 'search_image', 'import'
        self.params = {}
        self.collection = None
        self.db_path = "./chroma_db_store"
        self.collection_name = "engineering_components_v1"

    def run(self):
        if self.task == "init":
            self._init_db()
        elif self.task == "search_text":
            self._search_text()
        elif self.task == "search_image":
            self._search_image()
        elif self.task == "import":
            self._import_images()

    def _init_db(self):
        try:
            self.status_signal.emit("INFO", "MAGI SYSTEM INITIALIZING...")
            client = chromadb.PersistentClient(path=self.db_path)
            embedding_func = OpenCLIPEmbeddingFunction()
            self.collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_func,
                data_loader=ImageLoader(),
            )
            count = self.collection.count()
            self.status_signal.emit("SUCCESS", f"SYSTEM READY. INDEXED NODES: {count}")
        except Exception as e:
            self.status_signal.emit("ERROR", f"INIT FAILED: {str(e)}")

    def _search_text(self):
        try:
            query = self.params["query"]
            limit = self.params.get("limit", 10)
            self.status_signal.emit("INFO", f"ANALYZING PATTERN: {query}")

            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                include=["metadatas", "distances", "uris"],
            )
            self._process_results(results)
        except Exception as e:
            self.status_signal.emit("ERROR", f"SEARCH ERROR: {str(e)}")

    def _search_image(self):
        try:
            img_path = self.params["path"]
            limit = self.params.get("limit", 10)
            self.status_signal.emit("INFO", "VISUAL PATTERN MATCHING...")

            img_array = np.array(Image.open(img_path))
            results = self.collection.query(
                query_images=[img_array],
                n_results=limit,
                include=["metadatas", "distances", "uris"],
            )
            self._process_results(results)
        except Exception as e:
            self.status_signal.emit("ERROR", f"IMAGE SCAN ERROR: {str(e)}")

    def _process_results(self, results):
        formatted = []
        if results and results["ids"]:
            ids = results["ids"][0]
            dists = results["distances"][0]
            metas = results["metadatas"][0]
            uris = results["uris"][0]

            for i in range(len(ids)):
                formatted.append(
                    {
                        "id": ids[i],
                        "score": 1 - dists[i],
                        "meta": metas[i],
                        "uri": uris[i],
                    }
                )
        self.search_result_signal.emit(formatted)
        self.status_signal.emit("SUCCESS", f"FOUND {len(formatted)} MATCHING TARGETS")

    def _import_images(self):
        files = self.params["files"]  # list of paths
        total = len(files)
        self.status_signal.emit("INFO", f"STARTING BATCH IMPORT: {total} FILES")

        batch_ids, batch_uris, batch_metas = [], [], []
        BATCH_SIZE = 20

        for idx, fpath in enumerate(files):
            try:
                p = Path(fpath)
                # 簡單 Metadata (可擴充)
                meta = {
                    "filename": p.name,
                    "filepath": str(p),
                    "timestamp": str(time.time()),
                }
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(p)))

                batch_ids.append(uid)
                batch_uris.append(str(p))
                batch_metas.append(meta)

                if len(batch_ids) >= BATCH_SIZE:
                    self.collection.add(
                        ids=batch_ids, uris=batch_uris, metadatas=batch_metas
                    )
                    batch_ids, batch_uris, batch_metas = [], [], []
                    self.import_progress_signal.emit(idx + 1, total)

            except Exception as e:
                print(f"Skipping {fpath}: {e}")

        # 處理剩餘
        if batch_ids:
            self.collection.add(ids=batch_ids, uris=batch_uris, metadatas=batch_metas)

        self.import_progress_signal.emit(total, total)
        self.status_signal.emit("SUCCESS", "DATA SYNCHRONIZATION COMPLETE")


# ==========================================
# 3. 主視窗 (Command Center)
# ==========================================


class ResultWidget(EvaFrame):
    """單個搜尋結果卡片"""

    def __init__(self, data):
        super().__init__(color=NERV.COLOR_DIM)
        self.setFixedSize(220, 260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 圖片
        self.img_lbl = QLabel()
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setStyleSheet(
            f"border: 1px solid {NERV.COLOR_MAIN}; background: #000;"
        )

        pix = QPixmap(data["uri"])
        if not pix.isNull():
            # 保持比例縮放
            self.img_lbl.setPixmap(
                pix.scaled(200, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.img_lbl.setText("NO VISUAL")

        # 數據
        sync_rate = data["score"] * 100
        color = NERV.COLOR_ACCENT if sync_rate > 70 else NERV.COLOR_MAIN

        info = QLabel(
            f"SYNC RATE: {sync_rate:.1f}%\nID: {data['meta'].get('filename', 'UNK')[:15]}..."
        )
        info.setStyleSheet(
            f"color: {color}; font-family: '{NERV.FONT_TECH}'; font-size: 11px;"
        )

        layout.addWidget(self.img_lbl)
        layout.addWidget(info)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NERV // MAGI DATABASE INTERFACE")
        self.resize(1280, 800)

        # 初始化 Worker
        self.worker = DatabaseWorker()
        self.worker.status_signal.connect(self.log_status)
        self.worker.search_result_signal.connect(self.display_results)
        self.worker.import_progress_signal.connect(self.update_import_progress)

        # 狀態變數
        self.current_results_count = 0
        self.last_query_type = None  # 'text' or 'image'
        self.last_query_value = None

        self.init_ui()

        # 啟動 DB
        self.worker.task = "init"
        self.worker.start()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # --- 左側操作面板 (Tactical Ops) ---
        side_panel = QWidget()
        side_panel.setFixedWidth(350)
        side_panel.setStyleSheet(
            f"background: rgba(20, 20, 20, 0.9); border-right: 2px solid {NERV.COLOR_MAIN};"
        )
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(20, 20, 20, 20)

        # 1. Header
        header = QLabel("MAGI\nSYSTEM")
        header.setStyleSheet(
            f"font-family: '{NERV.FONT_HEADER}'; font-size: 48px; color: {NERV.COLOR_MAIN};"
        )
        header.setAlignment(Qt.AlignLeft)

        # 2. Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("border: none;")

        # Tab 1: 檢索 (Search)
        tab_search = QWidget()
        ts_layout = QVBoxLayout(tab_search)

        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("輸入特徵關鍵字...")
        self.input_text.returnPressed.connect(self.on_text_search)

        self.drop_zone = DropZone()
        self.drop_zone.setFixedHeight(120)
        self.drop_zone.fileDropped.connect(self.on_image_dropped)

        btn_search_text = EvaButton("發射 // 文字檢索")
        btn_search_text.clicked.connect(self.on_text_search)

        btn_search_img = EvaButton("掃描 // 視覺比對")
        btn_search_img.clicked.connect(self.on_image_search)

        ts_layout.addWidget(QLabel(":: TEXT INPUT ::"))
        ts_layout.addWidget(self.input_text)
        ts_layout.addWidget(btn_search_text)
        ts_layout.addSpacing(20)
        ts_layout.addWidget(QLabel(":: VISUAL INPUT ::"))
        ts_layout.addWidget(self.drop_zone)
        ts_layout.addWidget(btn_search_img)
        ts_layout.addStretch()

        # Tab 2: 入庫 (Import)
        tab_import = QWidget()
        ti_layout = QVBoxLayout(tab_import)

        btn_sel_files = EvaButton("選擇檔案 (Files)")
        btn_sel_files.clicked.connect(self.select_files)

        btn_sel_folder = EvaButton("選擇資料夾 (Folder)")
        btn_sel_folder.clicked.connect(self.select_folder)

        self.import_status = QLabel("等待指令...")
        self.import_status.setStyleSheet(
            f"color: {NERV.COLOR_MAIN}; font-family: '{NERV.FONT_TECH}';"
        )

        self.import_progress = QProgressBar()
        self.import_progress.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid {NERV.COLOR_MAIN}; background: #000; height: 10px; text-align: center; }}
            QProgressBar::chunk {{ background-color: {NERV.COLOR_MAIN}; }}
        """)

        ti_layout.addWidget(QLabel(":: DATA ENTRY ::"))
        ti_layout.addWidget(btn_sel_files)
        ti_layout.addWidget(btn_sel_folder)
        ti_layout.addSpacing(20)
        ti_layout.addWidget(self.import_status)
        ti_layout.addWidget(self.import_progress)
        ti_layout.addStretch()

        self.tabs.addTab(tab_search, "檢索模式")
        self.tabs.addTab(tab_import, "資料同步")

        # 3. Log
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet(
            f"background: transparent; border: 1px dashed {NERV.COLOR_MAIN}; color: {NERV.COLOR_MAIN}; font-family: '{NERV.FONT_TECH}'; font-size: 10px;"
        )
        self.log_box.setFixedHeight(150)

        side_layout.addWidget(header)
        side_layout.addWidget(self.tabs)
        side_layout.addWidget(self.log_box)

        # --- 右側結果顯示 (Visual Display) ---
        main_disp = QWidget()
        disp_layout = QVBoxLayout(main_disp)
        disp_layout.setContentsMargins(20, 20, 20, 20)

        # Top Bar
        top_bar = QHBoxLayout()
        self.status_lbl = QLabel("STATUS: STANDBY")
        self.status_lbl.setStyleSheet(
            f"font-family: '{NERV.FONT_HEADER}'; font-size: 24px; color: {NERV.COLOR_MAIN};"
        )

        btn_more = EvaButton("載入更多 // MORE", color=NERV.COLOR_ACCENT)
        btn_more.setFixedWidth(200)
        btn_more.clicked.connect(self.load_more)

        top_bar.addWidget(self.status_lbl)
        top_bar.addStretch()
        top_bar.addWidget(btn_more)

        # Grid Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_grid = QGridLayout(self.result_container)
        self.result_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        scroll.setWidget(self.result_container)

        disp_layout.addLayout(top_bar)
        disp_layout.addWidget(scroll)

        # Add to main
        main_layout.addWidget(side_panel)
        main_layout.addWidget(main_disp)

    # --- Logic Functions ---

    def log_status(self, level, msg):
        timestamp = time.strftime("%H:%M:%S")
        color = (
            NERV.COLOR_DANGER
            if level == "ERROR"
            else NERV.COLOR_ACCENT
            if level == "SUCCESS"
            else NERV.COLOR_MAIN
        )

        html = f"<span style='color:{color}'>[{timestamp}] {level}: {msg}</span>"
        self.log_box.appendHtml(html)

        self.status_lbl.setText(f"STATUS: {level} // {msg.split(':')[0]}")
        if level == "ERROR":
            self.status_lbl.setStyleSheet(
                f"font-family: '{NERV.FONT_HEADER}'; font-size: 24px; color: {NERV.COLOR_DANGER};"
            )
        else:
            self.status_lbl.setStyleSheet(
                f"font-family: '{NERV.FONT_HEADER}'; font-size: 24px; color: {NERV.COLOR_MAIN};"
            )

    def on_image_dropped(self, fpath):
        self.current_img_path = fpath

        # Update Preview in DropZone
        formatted_path = fpath.replace("\\", "/")
        self.drop_zone.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {NERV.COLOR_ACCENT};
                background-image: url({formatted_path}); 
                background-position: center;
                background-repeat: no-repeat;
                color: transparent;
            }}
        """)
        self.log_status("INFO", f"IMAGE LOADED: {Path(fpath).name}")

    def on_text_search(self):
        txt = self.input_text.text().strip()
        if not txt:
            return
        self.reset_results()
        self.last_query_type = "text"
        self.last_query_value = txt
        self.do_search(10)

    def on_image_search(self):
        if not hasattr(self, "current_img_path"):
            return
        self.reset_results()
        self.last_query_type = "image"
        self.last_query_value = self.current_img_path
        self.do_search(10)

    def load_more(self):
        if not self.last_query_type:
            return
        new_limit = self.current_results_count + 10
        self.do_search(new_limit)

    def do_search(self, limit):
        self.worker.params = {
            "query": self.last_query_value if self.last_query_type == "text" else None,
            "path": self.last_query_value if self.last_query_type == "image" else None,
            "limit": limit,
        }
        self.worker.task = f"search_{self.last_query_type}"
        self.worker.start()

    def reset_results(self):
        # Clear grid
        while self.result_grid.count():
            item = self.result_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.current_results_count = 0

    def display_results(self, results):
        self.reset_results()  # Simplified: clear and redraw all for sorting order
        self.current_results_count = len(results)

        row, col = 0, 0
        cols_per_row = 4

        for data in results:
            card = ResultWidget(data)
            self.result_grid.addWidget(card, row, col)
            col += 1
            if col >= cols_per_row:
                col = 0
                row += 1

        self.log_status("SUCCESS", "DISPLAY UPDATED")

    # --- Import Logic ---

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if files:
            self.start_import(files)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder:
            files = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                files.extend([str(p) for p in Path(folder).rglob(ext)])
            self.start_import(files)

    def start_import(self, files):
        if not files:
            return
        self.import_progress.setValue(0)
        self.worker.params = {"files": files}
        self.worker.task = "import"
        self.worker.start()

    def update_import_progress(self, current, total):
        self.import_progress.setValue(int(current / total * 100))
        self.import_status.setText(f"SYNCHRONIZING: {current}/{total}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)

    # 全域字體設定 (嘗試設定)
    font = QFont("Microsoft JhengHei UI", 10)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
