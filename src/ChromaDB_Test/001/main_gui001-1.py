import sys
import os
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QFileDialog,
    QScrollArea,
    QFrame,
    QGridLayout,
    QProgressBar,
    QTabWidget,
    QGraphicsOpacityEffect,
    QSizePolicy,
    QPlainTextEdit,
    QLabel,
    QPushButton
)
from PySide6.QtCore import (
    Qt,
    QThread,
    Signal,
    QSize,
    QPropertyAnimation,
    QEasingCurve,
    QTimer,
    QPoint,
    Property,
    QEvent,
)
from PySide6.QtGui import (
    QPixmap,
    QColor,
    QFont,
    QPainter,
    QPen,
    QBrush,
    QPainterPath,
    QDragEnterEvent,
    QDropEvent,
)

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# ==========================================
# 0. EVA 介面主題定義
# ==========================================
class EVA:
    COLOR_BG = "#0a0a0f"          # 深色背景
    COLOR_PANEL = "#11141d"       # 面板底色
    COLOR_ORANGE = "#ff8b00"      # EVA 橘
    COLOR_GREEN = "#6bffb5"       # 同步綠
    COLOR_TEXT = "#f2f2f2"        # 文字
    COLOR_DIM = "rgba(255, 139, 0, 0.45)"
    COLOR_LINE = "#1f2a3a"
    FONT_MAIN = "Microsoft JhengHei UI"
    FONT_TECH = "Consolas"
    FONT_HEADER = "Impact"


STYLESHEET = f"""
QMainWindow {{
    background-color: {EVA.COLOR_BG};
}}

QTabWidget::pane {{
    border: 2px solid {EVA.COLOR_ORANGE};
    background: rgba(10,10,15,0.8);
}}
QTabBar::tab {{
    background: #000;
    color: {EVA.COLOR_ORANGE};
    border: 1px solid {EVA.COLOR_ORANGE};
    padding: 8px 18px;
    margin-right: 4px;
    font-family: '{EVA.FONT_HEADER}';
    letter-spacing: 2px;
}}
QTabBar::tab:selected {{
    background: {EVA.COLOR_ORANGE};
    color: #000;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: #0f1622;
    width: 12px;
}}
QScrollBar::handle:vertical {{
    background: {EVA.COLOR_ORANGE};
    min-height: 20px;
}}

QLineEdit {{
    background: rgba(0,0,0,0.7);
    border: 1px solid {EVA.COLOR_ORANGE};
    color: {EVA.COLOR_GREEN};
    font-family: '{EVA.FONT_TECH}';
    font-size: 15px;
    padding: 8px 10px;
}}
QLineEdit:focus {{
    border: 2px solid {EVA.COLOR_GREEN};
}}

QProgressBar {{
    border: 1px solid {EVA.COLOR_ORANGE};
    background: #000;
    height: 12px;
    text-align: center;
    color: {EVA.COLOR_TEXT};
}}
QProgressBar::chunk {{
    background-color: {EVA.COLOR_ORANGE};
}}

QPlainTextEdit {{
    background: rgba(0,0,0,0.4);
    border: 1px dashed {EVA.COLOR_ORANGE};
    color: {EVA.COLOR_ORANGE};
    font-family: '{EVA.FONT_TECH}';
}}

QLabel#Title {{
    font-family: '{EVA.FONT_HEADER}';
    font-size: 42px;
    color: {EVA.COLOR_ORANGE};
    letter-spacing: 4px;
}}
QLabel#Subtitle {{
    font-family: '{EVA.FONT_TECH}';
    font-size: 12px;
    color: {EVA.COLOR_GREEN};
    letter-spacing: 3px;
}}
"""

# ==========================================
# 1. 工具與自訂元件
# ==========================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def get_image_metadata(file_path: Path, root_path: Path) -> Dict[str, Any]:
    """
    依據資料夾結構產生基礎 metadata，方便後續搜尋說明。
    """
    relative_path = file_path.relative_to(root_path) if root_path in file_path.parents else file_path.name
    parts = relative_path.parts if isinstance(relative_path, Path) else (relative_path,)

    metadata = {
        "filepath": str(file_path),
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "category": "未知",
        "part_id": "未知",
        "source_folder": str(root_path) if isinstance(root_path, Path) else str(file_path.parent),
    }

    if isinstance(parts, tuple) and len(parts) >= 2:
        metadata["category"] = parts[0]
        metadata["part_id"] = parts[1]

    return metadata


class EvaFrame(QFrame):
    """
    EVA 切角框，畫出向量線條營造科技感。
    """

    def __init__(self, color: str = EVA.COLOR_ORANGE, parent=None):
        super().__init__(parent)
        self.color_hex = color
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        w, h = rect.width(), rect.height()
        cut = 14

        path = QPainterPath()
        path.moveTo(rect.left() + cut, rect.top())
        path.lineTo(rect.right() - cut, rect.top())
        path.lineTo(rect.right(), rect.top() + cut)
        path.lineTo(rect.right(), rect.bottom() - cut)
        path.lineTo(rect.right() - cut, rect.bottom())
        path.lineTo(rect.left() + cut, rect.bottom())
        path.lineTo(rect.left(), rect.bottom() - cut)
        path.lineTo(rect.left(), rect.top() + cut)
        path.closeSubpath()

        fill_color = QColor(self.color_hex)
        fill_color.setAlpha(35)
        painter.fillPath(path, fill_color)

        pen = QPen(QColor(self.color_hex))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPath(path)

        # 內部十字線
        inner_pen = QPen(QColor(self.color_hex))
        inner_pen.setStyle(Qt.DotLine)
        inner_pen.setWidth(1)
        painter.setPen(inner_pen)
        painter.drawLine(rect.center().x(), rect.top() + 8, rect.center().x(), rect.bottom() - 8)
        painter.drawLine(rect.left() + 8, rect.center().y(), rect.right() - 8, rect.center().y())


class EvaButton(QPushButton):
    """
    帶掃描線與亮度動畫的 EVA 風格按鈕。
    """

    def __init__(self, text: str, color: str = EVA.COLOR_ORANGE, parent=None):
        super().__init__(text, parent)
        self.base_color = color
        self.hover_progress = 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.setMinimumHeight(42)
        self.setFont(QFont(EVA.FONT_HEADER, 12))
        self.setStyleSheet("border: none; background: transparent; color: #000;")

        self.anim = QPropertyAnimation(self, b"hover_progress_prop")
        self.anim.setDuration(260)
        self.anim.setEasingCurve(QEasingCurve.OutCubic)

    def set_hover_progress(self, val: float):
        self.hover_progress = val
        self.update()

    def get_hover_progress(self) -> float:
        return self.hover_progress

    hover_progress_prop = Property(float, get_hover_progress, set_hover_progress)

    def enterEvent(self, event):
        self.anim.stop()
        self.anim.setStartValue(self.hover_progress)
        self.anim.setEndValue(1.0)
        self.anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.anim.stop()
        self.anim.setStartValue(self.hover_progress)
        self.anim.setEndValue(0.0)
        self.anim.start()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect()
        w, h = rect.width(), rect.height()
        cut = 10

        current_color = QColor(EVA.COLOR_GREEN if self.hover_progress > 0.3 else self.base_color)
        bg_alpha = int(70 + 120 * self.hover_progress)
        fill_color = QColor(current_color)
        fill_color.setAlpha(bg_alpha)

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

        painter.setBrush(QBrush(fill_color))
        painter.setPen(QPen(current_color, 2))
        painter.drawPath(path)

        if self.hover_progress > 0:
            scan_y = int(h * self.hover_progress)
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
            painter.drawLine(0, scan_y, w, scan_y)

        painter.setPen(QPen(Qt.black if self.hover_progress > 0.4 else current_color, 2))
        painter.setFont(self.font())
        painter.drawText(rect, Qt.AlignCenter, self.text())


class DropZone(EvaFrame):
    """
    可拖拉檔案的區塊，會持續呼吸發光。
    """

    filesDropped = Signal(list)

    def __init__(self, text: str, color: str = EVA.COLOR_ORANGE, parent=None):
        super().__init__(color=color, parent=parent)
        self.setAcceptDrops(True)
        self.label = QLabel(text, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"color: {color}; font-family: '{EVA.FONT_HEADER}'; font-size: 14px;")
        self.pulse = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate_border)
        self.timer.start(120)

    def resizeEvent(self, event):
        self.label.setGeometry(self.rect())
        super().resizeEvent(event)

    def _animate_border(self):
        self.pulse = (self.pulse + 1) % 20
        alpha = 60 + int(80 * abs(10 - self.pulse) / 10)
        self.color_hex = f"rgba(255, 139, 0, {alpha / 255:.2f})"
        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.label.setText("連結建立，放開即可匯入")
            self.label.setStyleSheet(f"color: {EVA.COLOR_GREEN}; font-family: '{EVA.FONT_HEADER}';")
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not event.mimeData().hasUrls():
            return
        paths = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                paths.append(local)
        if paths:
            self.filesDropped.emit(paths)
            self.label.setText("資料已接收")
        else:
            self.label.setText("未偵測到檔案")


class PreviewBox(EvaFrame):
    """
    保持比例的圖片預覽框，附帶淡入動畫。
    """

    def __init__(self, parent=None):
        super().__init__(color=EVA.COLOR_DIM, parent=parent)
        self.setMinimumHeight(160)
        self.img_label = QLabel(self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background: #000;")
        self._pixmap = None

        self.fade_effect = QGraphicsOpacityEffect(self.img_label)
        self.img_label.setGraphicsEffect(self.fade_effect)
        self.fade_anim = QPropertyAnimation(self.fade_effect, b"opacity")
        self.fade_anim.setDuration(300)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)

    def resizeEvent(self, event):
        padding = 12
        self.img_label.setGeometry(
            padding, padding, self.width() - 2 * padding, self.height() - 2 * padding
        )
        self._refresh_pixmap()
        super().resizeEvent(event)

    def set_image(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self.img_label.setText("無法載入圖片")
            self._pixmap = None
            return
        self._pixmap = pix
        self._refresh_pixmap()
        self.fade_anim.stop()
        self.fade_effect.setOpacity(0.0)
        self.fade_anim.start()

    def _refresh_pixmap(self):
        if not self._pixmap:
            return
        target_size = QSize(self.img_label.width(), self.img_label.height())
        scaled = self._pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled)


class ResultCard(EvaFrame):
    """
    EVA 風格搜尋結果卡片，保持圖片比例並顯示同步率。
    """

    def __init__(self, data: Dict[str, Any]):
        super().__init__(color=EVA.COLOR_DIM)
        self.setFixedSize(230, 270)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet(f"background: #000; border: 1px solid {EVA.COLOR_LINE};")
        self.img_label.setFixedHeight(150)

        pix = QPixmap(data.get("uri", ""))
        if not pix.isNull():
            self.img_label.setPixmap(
                pix.scaled(QSize(210, 140), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.img_label.setText("影像遺失")
            self.img_label.setStyleSheet(f"color: {EVA.COLOR_ORANGE};")

        meta = data.get("meta", {})
        sync_rate = data.get("score", 0) * 100
        sync_label = QLabel(f"同步率：{sync_rate:.1f}%")
        sync_label.setStyleSheet(
            f"color: {EVA.COLOR_GREEN if sync_rate > 65 else EVA.COLOR_ORANGE};"
            f"font-family: '{EVA.FONT_TECH}';"
        )

        name = meta.get("filename", "未命名")
        part_id = meta.get("part_id", "未知")
        info_label = QLabel(f"{name}\n零件編碼：{part_id}")
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {EVA.COLOR_TEXT}; font-size: 12px;")

        layout.addWidget(self.img_label)
        layout.addWidget(sync_label)
        layout.addWidget(info_label)
        layout.addStretch()


# ==========================================
# 2. 後端 Worker
# ==========================================
class DatabaseWorker(QThread):
    status_signal = Signal(str, str)  # level, message
    search_result_signal = Signal(list)
    import_progress_signal = Signal(int, int)  # current, total

    def __init__(self):
        super().__init__()
        self.task = None  # 'init', 'search_text', 'search_image', 'import'
        self.params: Dict[str, Any] = {}
        self.collection = None
        self.db_path = "./chroma_db_store"
        self.collection_name = "engineering_components_v1"

    def ensure_collection(self) -> bool:
        if self.collection:
            return True
        try:
            client = chromadb.PersistentClient(path=self.db_path)
            embedding_func = OpenCLIPEmbeddingFunction()
            self.collection = client.get_or_create_collection(
                name=self.collection_name, embedding_function=embedding_func, data_loader=ImageLoader()
            )
            return True
        except Exception as e:
            self.status_signal.emit("錯誤", f"資料庫連線失敗：{e}")
            return False

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
        if not self.ensure_collection():
            return
        try:
            count = self.collection.count()
            self.status_signal.emit("連線", f"資料庫完成同步，現有向量：{count} 筆")
        except Exception as e:
            self.status_signal.emit("錯誤", f"初始化失敗：{e}")

    def _search_text(self):
        if not self.ensure_collection():
            return
        try:
            query = self.params.get("query", "")
            limit = self.params.get("limit", 12)
            self.status_signal.emit("搜尋", f"語意分析中：{query}")

            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                include=["metadatas", "distances", "uris"],
            )
            self._process_results(results)
        except Exception as e:
            self.status_signal.emit("錯誤", f"文字搜尋失敗：{e}")

    def _search_image(self):
        if not self.ensure_collection():
            return
        try:
            img_path = self.params.get("path", "")
            limit = self.params.get("limit", 12)
            self.status_signal.emit("搜尋", "視覺向量比對中…")

            img_array = np.array(Image.open(img_path))
            results = self.collection.query(
                query_images=[img_array],
                n_results=limit,
                include=["metadatas", "distances", "uris"],
            )
            self._process_results(results)
        except Exception as e:
            self.status_signal.emit("錯誤", f"圖片搜尋失敗：{e}")

    def _process_results(self, results):
        formatted = []
        if results and results.get("ids"):
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
        self.status_signal.emit("完成", f"取得 {len(formatted)} 筆結果")

    def _import_images(self):
        if not self.ensure_collection():
            return
        files: List[str] = self.params.get("files", [])
        root_param = self.params.get("root")
        total = len(files)
        if total == 0:
            self.status_signal.emit("提示", "沒有可匯入的圖片")
            return

        self.status_signal.emit("入庫", f"開始同步 {total} 張圖片")
        batch_ids: List[str] = []
        batch_uris: List[str] = []
        batch_metas: List[Dict[str, Any]] = []
        BATCH_SIZE = 24

        for idx, fpath in enumerate(files):
            try:
                p = Path(fpath)
                root_path = Path(root_param) if root_param else p.parent
                meta = get_image_metadata(p, root_path)
                uid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(p)))

                batch_ids.append(uid)
                batch_uris.append(str(p))
                batch_metas.append(meta)

                if len(batch_ids) >= BATCH_SIZE:
                    self.collection.add(ids=batch_ids, uris=batch_uris, metadatas=batch_metas)
                    batch_ids, batch_uris, batch_metas = [], [], []
                    self.import_progress_signal.emit(idx + 1, total)
            except Exception as e:
                self.status_signal.emit("警告", f"略過 {fpath}: {e}")

        if batch_ids:
            self.collection.add(ids=batch_ids, uris=batch_uris, metadatas=batch_metas)

        self.import_progress_signal.emit(total, total)
        self.status_signal.emit("完成", "圖片已寫入資料庫")


# ==========================================
# 3. 主介面
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EVA 神經資料庫 // 影像檢索艦橋")
        self.resize(1320, 860)

        self.worker = DatabaseWorker()
        self.worker.status_signal.connect(self.log_status)
        self.worker.search_result_signal.connect(self.display_results)
        self.worker.import_progress_signal.connect(self.update_import_progress)

        self.last_query_type = None  # 'text' or 'image'
        self.last_query_value = None
        self.current_limit = 12
        self.current_img_path = None

        self._build_ui()
        self._apply_intro_animation()

        self.worker.task = "init"
        self.worker.start()

    # ---------- UI 組裝 ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        side_panel = self._build_side_panel()
        result_panel = self._build_result_panel()

        main_layout.addWidget(side_panel)
        main_layout.addWidget(result_panel, 1)

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(380)
        panel.setStyleSheet(f"background: {EVA.COLOR_PANEL}; border: 2px solid {EVA.COLOR_LINE};")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("EVA 知識終端")
        title.setObjectName("Title")
        subtitle = QLabel("神經連線・零件影像資料庫")
        subtitle.setObjectName("Subtitle")

        layout.addWidget(title)
        layout.addWidget(subtitle)

        tabs = QTabWidget()
        tabs.addTab(self._build_search_tab(), "檢索")
        tabs.addTab(self._build_import_tab(), "入庫")
        layout.addWidget(tabs)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(150)
        layout.addWidget(self.log_box)

        return panel

    def _build_search_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        label_text = QLabel("文字檢索")
        label_text.setStyleSheet(f"color: {EVA.COLOR_TEXT}; font-family: '{EVA.FONT_HEADER}';")
        self.input_text = QLineEdit()
        self.input_text.setPlaceholderText("輸入要搜尋的零件描述、型號、特徵…")
        self.input_text.returnPressed.connect(self.on_text_search)
        self._attach_focus_glow(self.input_text)

        btn_text = EvaButton("立即搜尋")
        btn_text.clicked.connect(self.on_text_search)

        label_img = QLabel("以圖搜圖")
        label_img.setStyleSheet(f"color: {EVA.COLOR_TEXT}; font-family: '{EVA.FONT_HEADER}';")

        self.search_drop = DropZone("拖曳圖片到此")
        self.search_drop.setFixedHeight(110)
        self.search_drop.filesDropped.connect(self.on_search_drop)

        self.preview_box = PreviewBox()

        btn_pick_img = EvaButton("選取圖片")
        btn_pick_img.clicked.connect(self.choose_search_image)

        btn_img_search = EvaButton("視覺掃描", color=EVA.COLOR_GREEN)
        btn_img_search.clicked.connect(self.on_image_search)

        layout.addWidget(label_text)
        layout.addWidget(self.input_text)
        layout.addWidget(btn_text)
        layout.addSpacing(10)
        layout.addWidget(label_img)
        layout.addWidget(self.search_drop)
        layout.addWidget(self.preview_box)
        layout.addWidget(btn_pick_img)
        layout.addWidget(btn_img_search)
        layout.addStretch()

        return tab

    def _build_import_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(12)

        info = QLabel("資料入庫")
        info.setStyleSheet(f"color: {EVA.COLOR_TEXT}; font-family: '{EVA.FONT_HEADER}';")

        self.import_drop = DropZone("拖曳資料夾或多張圖片到此")
        self.import_drop.setFixedHeight(110)
        self.import_drop.filesDropped.connect(self.on_import_drop)

        btn_files = EvaButton("選擇圖片檔")
        btn_files.clicked.connect(self.select_files)
        btn_folder = EvaButton("選擇資料夾", color=EVA.COLOR_GREEN)
        btn_folder.clicked.connect(self.select_folder)

        self.import_status = QLabel("等待指令…")
        self.import_status.setStyleSheet(
            f"color: {EVA.COLOR_ORANGE}; font-family: '{EVA.FONT_TECH}'; font-size: 12px;"
        )

        self.import_progress = QProgressBar()

        note = QLabel("支援單張、多張或整個資料夾，會自動解析資料夾結構建立標籤。")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {EVA.COLOR_TEXT}; font-size: 12px;")

        layout.addWidget(info)
        layout.addWidget(self.import_drop)
        layout.addWidget(btn_files)
        layout.addWidget(btn_folder)
        layout.addWidget(self.import_status)
        layout.addWidget(self.import_progress)
        layout.addWidget(note)
        layout.addStretch()

        return tab

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        top_bar = QHBoxLayout()
        self.status_lbl = QLabel("狀態：待命")
        self.status_lbl.setStyleSheet(
            f"font-family: '{EVA.FONT_HEADER}'; font-size: 22px; color: {EVA.COLOR_ORANGE};"
        )

        self.more_btn = EvaButton("搜尋更多", color=EVA.COLOR_GREEN)
        self.more_btn.setFixedWidth(160)
        self.more_btn.clicked.connect(self.load_more)

        top_bar.addWidget(self.status_lbl)
        top_bar.addStretch()
        top_bar.addWidget(self.more_btn)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.result_container = QWidget()
        self.result_grid = QGridLayout(self.result_container)
        self.result_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_grid.setSpacing(12)
        scroll.setWidget(self.result_container)

        layout.addLayout(top_bar)
        layout.addWidget(scroll)
        return panel

    def _apply_intro_animation(self):
        effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(effect)
        self._intro_anim = QPropertyAnimation(effect, b"opacity")
        self._intro_anim.setDuration(900)
        self._intro_anim.setStartValue(0.0)
        self._intro_anim.setEndValue(1.0)
        self._intro_anim.start()

    def _attach_focus_glow(self, widget: QWidget):
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.9)
        widget.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", widget)
        anim.setDuration(220)
        anim.setStartValue(0.85)
        anim.setEndValue(1.0)
        widget._focus_anim = anim  # type: ignore
        widget.installEventFilter(self)

    # ---------- 事件處理 ----------
    def eventFilter(self, obj, event):
        if hasattr(obj, "_focus_anim"):
            if event.type() == QEvent.FocusIn:
                obj._focus_anim.setDirection(QPropertyAnimation.Forward)
                obj._focus_anim.start()
            elif event.type() == QEvent.FocusOut:
                obj._focus_anim.setDirection(QPropertyAnimation.Backward)
                obj._focus_anim.start()
        return super().eventFilter(obj, event)

    def log_status(self, level: str, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        color = (
            EVA.COLOR_GREEN
            if level in ["完成", "連線"]
            else EVA.COLOR_ORANGE
            if level in ["搜尋", "入庫", "提示"]
            else "#ff4d4f"
        )
        self.log_box.appendHtml(f"<span style='color:{color}'>[{timestamp}] {level}｜{msg}</span>")
        self.status_lbl.setText(f"狀態：{level}")
        self.status_lbl.setStyleSheet(
            f"font-family: '{EVA.FONT_HEADER}'; font-size: 22px; color: {color};"
        )

    def choose_search_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "選取圖片", ".", "圖片檔 (*.png *.jpg *.jpeg *.bmp *.webp)")
        if fname:
            self.current_img_path = fname
            self.preview_box.set_image(fname)
            self.search_drop.label.setText(Path(fname).name)
            self.log_status("搜尋", f"已選取圖片：{Path(fname).name}")

    def on_search_drop(self, paths: List[str]):
        images = self._collect_image_files(paths)
        if not images:
            self.log_status("提示", "拖曳內容中沒有可用圖片")
            return
        self.current_img_path = images[0]
        self.preview_box.set_image(images[0])
        self.log_status("搜尋", f"載入圖片：{Path(images[0]).name}")

    def on_import_drop(self, paths: List[str]):
        files = self._collect_image_files(paths, recursive=True)
        if not files:
            self.log_status("提示", "未偵測到可匯入的圖片")
            return
        root = self._determine_root(files)
        self.start_import(files, root=root)

    def on_text_search(self):
        text = self.input_text.text().strip()
        if not text:
            self.log_status("提示", "請輸入關鍵字")
            return
        self.last_query_type = "text"
        self.last_query_value = text
        self.current_limit = 12
        self._launch_search()

    def on_image_search(self):
        if not self.current_img_path:
            self.log_status("提示", "請先選擇或拖入圖片")
            return
        self.last_query_type = "image"
        self.last_query_value = self.current_img_path
        self.current_limit = 12
        self._launch_search()

    def load_more(self):
        if not self.last_query_type:
            self.log_status("提示", "尚未執行任何搜尋")
            return
        self.current_limit += 8
        self._launch_search()

    def _launch_search(self):
        if self.worker.isRunning():
            self.log_status("提示", "系統忙碌中，稍候再試")
            return

        self.clear_results()
        params = {"limit": self.current_limit}
        if self.last_query_type == "text":
            params["query"] = self.last_query_value
            self.worker.task = "search_text"
        else:
            params["path"] = self.last_query_value
            self.worker.task = "search_image"
        self.worker.params = params
        self.worker.start()
        self.status_lbl.setText("狀態：搜尋中…")

    def clear_results(self):
        while self.result_grid.count():
            item = self.result_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def display_results(self, results: List[Dict[str, Any]]):
        self.clear_results()
        if not results:
            empty = QLabel("沒有找到相符資料。")
            empty.setStyleSheet(f"color: {EVA.COLOR_TEXT}; font-size: 16px; padding: 20px;")
            self.result_grid.addWidget(empty, 0, 0)
            return

        row = col = 0
        per_row = 4
        for data in results:
            card = ResultCard(data)
            self.result_grid.addWidget(card, row, col)
            col += 1
            if col >= per_row:
                col = 0
                row += 1

    # ---------- 匯入功能 ----------
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "選擇圖片",
            ".",
            "圖片檔 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if files:
            root = self._determine_root(files)
            self.start_import(files, root=root)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if folder:
            files = self._collect_image_files([folder], recursive=True)
            root = folder
            self.start_import(files, root=root)

    def start_import(self, files: List[str], root: str = None):
        if not files:
            self.log_status("提示", "沒有可匯入的檔案")
            return
        if self.worker.isRunning():
            self.log_status("提示", "背景任務執行中，請稍候")
            return

        self.import_progress.setValue(0)
        self.import_status.setText("入庫中…")

        self.worker.params = {"files": files, "root": root}
        self.worker.task = "import"
        self.worker.start()

    def update_import_progress(self, current: int, total: int):
        if total == 0:
            return
        percent = int(current / total * 100)
        self.import_progress.setValue(percent)
        self.import_status.setText(f"入庫進度：{current}/{total}")

    # ---------- 輔助 ----------
    def _collect_image_files(self, paths: List[str], recursive: bool = False) -> List[str]:
        collected: List[str] = []
        for raw in paths:
            p = Path(raw)
            if p.is_dir() and recursive:
                for fp in p.rglob("*"):
                    if fp.suffix.lower() in IMAGE_EXTS:
                        collected.append(str(fp))
            elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                collected.append(str(p))
        return collected

    def _determine_root(self, files: List[str]) -> str:
        try:
            return os.path.commonpath(files)
        except Exception:
            return str(Path(files[0]).parent)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    app.setFont(QFont(EVA.FONT_MAIN, 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
