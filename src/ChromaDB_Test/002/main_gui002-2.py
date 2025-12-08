import os
import sys
from datetime import datetime

# ChromaDB 相關
import chromadb
import pandas as pd
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QPixmap,
)

# GUI 相關
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------
# Part 1: 後端邏輯封裝 (Backend Logic)
# ---------------------------------------------------------


class ChromaManager:
    """
    負責與 ChromaDB 溝通的後端類別。
    保持資料庫連線，避免重複初始化。
    """

    def __init__(self, persist_path="./chroma_data"):
        self.persist_path = persist_path
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embedding_func = OpenCLIPEmbeddingFunction()
        self.data_loader = ImageLoader()
        self.collection = self.client.get_or_create_collection(
            name="engineering_components_gallery",
            embedding_function=self.embedding_func,
            data_loader=self.data_loader,
        )

    def is_target_image(self, filename: str, item_name: str) -> bool:
        """過濾邏輯"""
        if not filename.lower().endswith(".png"):
            return False
        if filename == f"{item_name}_merged.png":
            return True
        prefix = f"{item_name}_random_"
        if filename.startswith(prefix):
            try:
                suffix = filename[len(prefix) : -4]
                num = int(suffix)
                if 1 <= num <= 20:
                    return True
            except ValueError:
                pass
        return False

    def ingest_folder(self, root_dir: str, progress_callback=None):
        """匯入資料夾"""
        documents = {"ids": [], "uris": [], "metadatas": []}

        # 簡單掃描檔案總數以便計算進度 (預估)
        total_files = sum([len(files) for r, d, files in os.walk(root_dir)])
        processed = 0

        for category in os.listdir(root_dir):
            cat_path = os.path.join(root_dir, category)
            if not os.path.isdir(cat_path):
                continue

            for item_name in os.listdir(cat_path):
                item_path = os.path.join(cat_path, item_name)
                if not os.path.isdir(item_path):
                    continue

                # 只掃描這一層
                try:
                    files = [
                        f
                        for f in os.listdir(item_path)
                        if os.path.isfile(os.path.join(item_path, f))
                    ]
                    for file in files:
                        processed += 1
                        if self.is_target_image(file, item_name):
                            full_path = os.path.join(item_path, file)
                            unique_id = f"{category}_{item_name}_{file}"

                            # 檢查是否已存在 (簡單增量更新)
                            existing = self.collection.get(ids=[unique_id])
                            if existing and len(existing["ids"]) > 0:
                                continue

                            documents["ids"].append(unique_id)
                            documents["uris"].append(full_path)
                            documents["metadatas"].append(
                                {
                                    "category": category,
                                    "item_name": item_name,
                                    "filename": file,
                                    "added_date": datetime.now().isoformat(),
                                    "file_path": full_path,
                                }
                            )

                        if progress_callback and processed % 10 == 0:
                            progress_callback(
                                int((processed / total_files) * 50)
                            )  # 掃描佔 50%
                except Exception as e:
                    print(f"Error accessing {item_path}: {e}")

        # 批次寫入
        count = len(documents["ids"])
        if count > 0:
            batch_size = 20
            for i in range(0, count, batch_size):
                end = min(i + batch_size, count)
                self.collection.add(
                    ids=documents["ids"][i:end],
                    uris=documents["uris"][i:end],
                    metadatas=documents["metadatas"][i:end],
                )
                if progress_callback:
                    # 寫入佔剩下的 50%
                    current_percent = 50 + int((i / count) * 50)
                    progress_callback(current_percent)

        return count

    def ingest_files(self, file_paths: list[str]):
        """匯入單張或多張圖片 (手動選取模式，Metadata 可能較少)"""
        ids, uris, metas = [], [], []
        for path in file_paths:
            filename = os.path.basename(path)
            # 嘗試從路徑推斷 item_name (假設遵循結構)，若無則標記為手動匯入
            parent_dir = os.path.dirname(path)
            item_name = os.path.basename(parent_dir)
            category = os.path.basename(os.path.dirname(parent_dir))

            unique_id = f"manual_{datetime.now().timestamp()}_{filename}"
            ids.append(unique_id)
            uris.append(path)
            metas.append(
                {
                    "category": category if category else "Manual_Import",
                    "item_name": item_name if item_name else "Manual_Import",
                    "filename": filename,
                    "file_path": path,
                    "added_date": datetime.now().isoformat(),
                }
            )

        if ids:
            self.collection.add(ids=ids, uris=uris, metadatas=metas)
        return len(ids)

    def search(self, query_path: str, top_k: int = 10):
        """搜尋並聚合結果"""
        results = self.collection.query(
            query_uris=[query_path], n_results=top_k, include=["metadatas", "distances"]
        )

        if not results["metadatas"] or not results["metadatas"][0]:
            return []

        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        parsed_data = []
        for meta, dist in zip(metadatas, distances):
            parsed_data.append(
                {
                    "item_name": meta.get("item_name", "Unknown"),
                    "category": meta.get("category", "Unknown"),
                    "similarity": 1 - dist,
                    "file_path": meta.get("file_path", ""),
                    "filename": meta.get("filename", ""),
                }
            )

        return parsed_data


# ---------------------------------------------------------
# Part 2: 自訂 UI 元件 (Custom Widgets with Neumorphism)
# ---------------------------------------------------------


class NeumorphicStyle:
    BG_COLOR = "#2b2b2b"
    PANEL_COLOR = "#323232"
    TEXT_COLOR = "#E0E0E0"
    ACCENT_COLOR = "#00d4ff"  # Cyan Glow
    SHADOW_LIGHT = "#3d3d3d"
    SHADOW_DARK = "#1f1f1f"
    FONT_FAMILY = "Microsoft JhengHei UI"

    @staticmethod
    def get_main_style():
        return f"""
            QMainWindow, QWidget {{
                background-color: {NeumorphicStyle.BG_COLOR};
                color: {NeumorphicStyle.TEXT_COLOR};
                font-family: '{NeumorphicStyle.FONT_FAMILY}';
                font-size: 14px;
            }}
            QScrollArea {{ border: none; background: transparent; }}
            QScrollBar:vertical {{
                background: {NeumorphicStyle.BG_COLOR};
                width: 10px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: #4a4a4a;
                min-height: 20px;
                border-radius: 5px;
            }}
        """


class NeumorphicButton(QPushButton):
    def __init__(self, text, parent=None, is_primary=False):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(45)

        base_color = "#383838" if not is_primary else "#006075"
        text_color = "#ffffff" if is_primary else "#e0e0e0"
        border_col = "#00d4ff" if is_primary else "#555555"

        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {base_color};
                color: {text_color};
                border-radius: 12px;
                border: 1px solid #404040;
                font-weight: bold;
                padding: 5px 15px;
            }}
            QPushButton:hover {{
                background-color: {"#007a94" if is_primary else "#454545"};
                border: 1px solid {border_col};
            }}
            QPushButton:pressed {{
                background-color: {"#004e5f" if is_primary else "#2a2a2a"};
                padding-top: 7px; 
                padding-left: 17px;
            }}
        """)

        # Add Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setOffset(4, 4)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)


class ImageDropLabel(QLabel):
    """支援拖拉圖片的 Label"""

    imageDropped = Signal(str)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setText("拖曳圖片至此\n或點擊選擇")
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {NeumorphicStyle.PANEL_COLOR};
                border: 2px dashed #555;
                border-radius: 20px;
                color: #888;
                font-size: 16px;
            }}
            QLabel:hover {{
                border-color: {NeumorphicStyle.ACCENT_COLOR};
                color: {NeumorphicStyle.ACCENT_COLOR};
            }}
        """)
        self.setAcceptDrops(True)
        self.setMinimumHeight(300)
        self.current_path = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                self.load_image(path)
                self.imageDropped.emit(path)

    def mousePressEvent(self, event):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.load_image(path)
            self.imageDropped.emit(path)

    def load_image(self, path):
        self.current_path = path
        pixmap = QPixmap(path)
        # 保持比例縮放
        scaled_pixmap = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
        # 更新 Style 移除 dashed border
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {NeumorphicStyle.PANEL_COLOR};
                border: 2px solid #444;
                border-radius: 20px;
            }}
        """)


class ResultCard(QFrame):
    """顯示單個搜尋結果的卡片"""

    def __init__(self, data):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {NeumorphicStyle.PANEL_COLOR};
                border-radius: 15px;
                border: 1px solid #3d3d3d;
            }}
        """)
        self.setFixedWidth(280)
        self.setFixedHeight(120)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 左側縮圖
        img_lbl = QLabel()
        img_lbl.setFixedSize(100, 100)
        img_lbl.setStyleSheet("background-color: #222; border-radius: 8px;")
        img_lbl.setAlignment(Qt.AlignCenter)

        if os.path.exists(data["file_path"]):
            pix = QPixmap(data["file_path"]).scaled(
                100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            img_lbl.setPixmap(pix)
        else:
            img_lbl.setText("遺失")

        # 右側資訊
        info_layout = QVBoxLayout()
        info_layout.setAlignment(Qt.AlignVCenter)

        name_lbl = QLabel(data["item_name"])
        name_lbl.setStyleSheet("font-weight: bold; font-size: 14px; color: white;")
        name_lbl.setWordWrap(True)

        cat_lbl = QLabel(f"分類: {data['category']}")
        cat_lbl.setStyleSheet("color: #aaa; font-size: 12px;")

        sim_val = data["similarity"] * 100
        sim_color = (
            "#00ff88" if sim_val > 90 else "#ffcc00" if sim_val > 70 else "#ff5555"
        )
        sim_lbl = QLabel(f"相似度: {sim_val:.1f}%")
        sim_lbl.setStyleSheet(
            f"color: {sim_color}; font-weight: bold; font-size: 13px;"
        )

        info_layout.addWidget(name_lbl)
        info_layout.addWidget(cat_lbl)
        info_layout.addWidget(sim_lbl)

        layout.addWidget(img_lbl)
        layout.addLayout(info_layout)

        # Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setOffset(3, 3)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)


# ---------------------------------------------------------
# Part 3: Worker Threads (防止介面卡頓)
# ---------------------------------------------------------


class SearchWorker(QThread):
    finished = Signal(object)  # 回傳聚合後的 DataFrame

    def __init__(self, manager, img_path, k):
        super().__init__()
        self.manager = manager
        self.img_path = img_path
        self.k = k

    def run(self):
        raw_results = self.manager.search(self.img_path, self.k)
        if not raw_results:
            self.finished.emit(None)
            return

        df = pd.DataFrame(raw_results)
        # 聚合計算
        stats = (
            df.groupby("item_name")
            .agg(
                category=("category", "first"),
                count=("item_name", "count"),
                max_similarity=("similarity", "max"),
                avg_similarity=("similarity", "mean"),
                sample_path=("file_path", "first"),  # 取一張圖當代表
            )
            .reset_index()
        )

        stats_sorted = stats.sort_values(
            by="max_similarity", ascending=False
        ).reset_index(drop=True)
        self.finished.emit(stats_sorted)


class IngestWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)

    def __init__(self, manager, paths, is_folder=False):
        super().__init__()
        self.manager = manager
        self.paths = paths
        self.is_folder = is_folder

    def run(self):
        try:
            count = 0
            if self.is_folder:
                count = self.manager.ingest_folder(self.paths[0], self.progress.emit)
            else:
                self.progress.emit(10)
                count = self.manager.ingest_files(self.paths)
                self.progress.emit(100)
            self.finished.emit(f"成功匯入 {count} 張圖片！")
        except Exception as e:
            self.finished.emit(f"匯入失敗: {str(e)}")


# ---------------------------------------------------------
# Part 4: Main Window
# ---------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 圖像檢索系統 | 鈑金零件辨識")
        self.resize(1200, 800)
        self.setStyleSheet(NeumorphicStyle.get_main_style())

        self.manager = ChromaManager()
        self.current_k = 20  # 初始搜尋數量
        self.last_query_path = None

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # === 左側面板：輸入與控制 ===
        left_panel = QFrame()
        left_panel.setFixedWidth(350)
        left_panel.setStyleSheet(
            f"background-color: {NeumorphicStyle.PANEL_COLOR}; border-radius: 20px;"
        )
        # Shadow for panel
        l_shadow = QGraphicsDropShadowEffect()
        l_shadow.setBlurRadius(20)
        l_shadow.setColor(QColor(0, 0, 0, 100))
        l_shadow.setOffset(0, 5)
        left_panel.setGraphicsEffect(l_shadow)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)
        left_layout.setContentsMargins(20, 30, 20, 30)

        # Title
        title_lbl = QLabel("檢索控制台")
        title_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #fff;")
        title_lbl.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title_lbl)

        # Image Drop Area
        self.drop_area = ImageDropLabel()
        self.drop_area.imageDropped.connect(self.on_image_selected)
        left_layout.addWidget(self.drop_area)

        # Filename Label
        self.filename_lbl = QLabel("- 尚未選擇圖片 -")
        self.filename_lbl.setAlignment(Qt.AlignCenter)
        self.filename_lbl.setStyleSheet("color: #888; font-style: italic;")
        left_layout.addWidget(self.filename_lbl)

        left_layout.addStretch()

        # Database Controls
        db_group_lbl = QLabel("資料庫管理")
        db_group_lbl.setStyleSheet(
            "color: #00d4ff; font-weight: bold; margin-top: 20px;"
        )
        left_layout.addWidget(db_group_lbl)

        btn_add_folder = NeumorphicButton("📂 匯入資料夾")
        btn_add_folder.clicked.connect(self.add_folder)
        left_layout.addWidget(btn_add_folder)

        btn_add_files = NeumorphicButton("📄 匯入圖片 (多選)")
        btn_add_files.clicked.connect(self.add_files)
        left_layout.addWidget(btn_add_files)

        # Progress Bar (Hidden initially)
        self.pbar = QProgressBar()
        self.pbar.setStyleSheet("""
            QProgressBar { border: none; background: #222; border-radius: 5px; height: 10px; text-align: center; }
            QProgressBar::chunk { background-color: #00d4ff; border-radius: 5px; }
        """)
        self.pbar.hide()
        left_layout.addWidget(self.pbar)

        main_layout.addWidget(left_panel)

        # === 右側面板：搜尋結果 ===
        right_layout = QVBoxLayout()

        # Header
        r_header_layout = QHBoxLayout()
        r_title = QLabel("辨識結果分析")
        r_title.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.result_status = QLabel("等待查詢...")
        self.result_status.setStyleSheet("color: #888;")
        r_header_layout.addWidget(r_title)
        r_header_layout.addStretch()
        r_header_layout.addWidget(self.result_status)
        right_layout.addLayout(r_header_layout)

        # Scroll Area for Results
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.results_grid = QVBoxLayout(
            self.scroll_content
        )  # 使用 VBox 條列式，或 FlowLayout 網格
        self.results_grid.setSpacing(15)
        self.results_grid.setAlignment(Qt.AlignTop)
        scroll.setWidget(self.scroll_content)
        right_layout.addWidget(scroll)

        # Search More Button
        self.btn_search_more = NeumorphicButton(
            "🔍 找不到預期結果？ 搜尋更多...", is_primary=True
        )
        self.btn_search_more.clicked.connect(self.search_more)
        self.btn_search_more.hide()
        right_layout.addWidget(self.btn_search_more)

        main_layout.addLayout(right_layout)

    # --- Logic Implementations ---

    def on_image_selected(self, path):
        self.last_query_path = path
        self.filename_lbl.setText(os.path.basename(path))
        self.current_k = 20  # Reset K
        self.start_search(path, self.current_k)

    def start_search(self, path, k):
        # 清空舊結果
        if k == 20:  # 若是新搜尋才清空
            self.clear_results()

        self.result_status.setText(f"正在分析... (Top {k})")
        self.btn_search_more.setEnabled(False)
        self.btn_search_more.setText("分析中...")

        self.search_thread = SearchWorker(self.manager, path, k)
        self.search_thread.finished.connect(self.on_search_finished)
        self.search_thread.start()

    def on_search_finished(self, df):
        self.clear_results()
        self.btn_search_more.setEnabled(True)
        self.btn_search_more.setText("🔍 找不到預期結果？ 搜尋更多...")
        self.btn_search_more.show()

        if df is None or df.empty:
            self.result_status.setText("未找到相似圖片")
            # Show empty state
            lbl = QLabel("資料庫中沒有找到相似的特徵。\n請嘗試匯入更多資料或檢查圖片。")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #666; font-size: 16px; margin-top: 50px;")
            self.results_grid.addWidget(lbl)
            return

        top_item = df.iloc[0]
        self.result_status.setText(
            f"最佳匹配: {top_item['item_name']} (信心: {top_item['max_similarity']:.2%})"
        )

        # 顯示卡片
        for _, row in df.iterrows():
            card = ResultCard(row)
            self.results_grid.addWidget(card)

    def search_more(self):
        if self.last_query_path:
            self.current_k += 20  # 增加搜尋範圍
            self.start_search(self.last_query_path, self.current_k)

    def clear_results(self):
        while self.results_grid.count():
            item = self.results_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def add_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if dir_path:
            self.start_ingest([dir_path], is_folder=True)

    def add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "選擇圖片", "", "Images (*.png *.jpg *.jpeg)"
        )
        if files:
            self.start_ingest(files, is_folder=False)

    def start_ingest(self, paths, is_folder):
        self.pbar.show()
        self.pbar.setValue(0)
        self.ingest_thread = IngestWorker(self.manager, paths, is_folder)
        self.ingest_thread.progress.connect(self.pbar.setValue)
        self.ingest_thread.finished.connect(self.on_ingest_finished)
        self.ingest_thread.start()

    def on_ingest_finished(self, msg):
        self.pbar.hide()
        QMessageBox.information(self, "匯入完成", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
