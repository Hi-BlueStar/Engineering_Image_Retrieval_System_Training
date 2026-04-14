"""
索引建立模組。
負責掃描來源 (檔案、目錄)、提取特徵並寫入 ChromaDB。
支援單檔、多檔及遞迴目錄掃描。
"""
import os
import cv2
import logging
from pathlib import Path
from typing import List, Union, Optional
from tqdm import tqdm
import numpy as np

from src.vector_search.database import ChromaDBManager
from src.vector_search.feature_extractor import SimSiamFeatureExtractor

# 引入影像前處理工具
# 引入共用工具
from src.vector_search.utils import extract_rois_from_image

logger = logging.getLogger(__name__)

class ImageIndexer:
    """
    影像索引建立器。
    """
    
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, db_manager: ChromaDBManager, feature_extractor: SimSiamFeatureExtractor):
        """
        初始化索引器。

        Args:
            db_manager: ChromaDB 管理器實例。
            feature_extractor: 特徵提取器實例。
        """
        self.db = db_manager
        self.extractor = feature_extractor

    def _collect_image_paths(self, sources: Union[str, List[str]]) -> List[Path]:
        """
        收集所有有效影像路徑。
        支援遞迴掃描目錄。
        """
        if isinstance(sources, str):
            sources = [sources]
            
        image_paths = []
        for src in sources:
            path = Path(src)
            if path.is_file():
                if path.suffix.lower() in self.VALID_EXTENSIONS:
                    image_paths.append(path)
            elif path.is_dir():
                # 遞迴掃描
                for p in path.rglob("*"):
                    if p.is_file() and p.suffix.lower() in self.VALID_EXTENSIONS:
                        image_paths.append(p)
            else:
                logger.warning(f"來源不存在或無效: {src}")
                
        # 去重
        return sorted(list(set(image_paths)))

    def index(self, sources: Union[str, List[str]], batch_size: int = 32) -> int:
        """
        執行索引建立流程。

        Args:
            sources (Union[str, List[str]]): 來源路徑 (檔案、目錄或列表)。
            batch_size (int): 批次處理大小，避免記憶體溢出。

        Returns:
            int: 成功索引的影像數量。
        """
        logger.info("開始掃描影像來源...")
        image_paths = self._collect_image_paths(sources)
        total_images = len(image_paths)
        
        if total_images == 0:
            logger.warning("未找到任何有效影像。")
            return 0
            
        logger.info(f"共找到 {total_images} 張影像，準備開始建立索引 (Batch Size: {batch_size})")

        success_count = 0
        
        # 批次處理
        for i in tqdm(range(0, total_images, batch_size), desc="Indexing"):
            batch_paths = image_paths[i : i + batch_size]
            
            # 1. 讀取影像
            images = []
            valid_paths = []
            
            for p in batch_paths:
                # cv2 imread 不支援非 ASCII 路徑，需用 imdecode (或檢查路徑是否全英文)
                # 簡單起見，先嘗試直接讀，若專案有 imread_unicode 工具應引用之
                # 這裡使用專案慣用的 cv2.imdecode 以支援中文路徑
                try:
                    # Python Open works with unicode, then numpy frombuffer to cv2
                    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append(img)
                        valid_paths.append(p)
                    else:
                        logger.warning(f"無法讀取影像 (可能損壞): {p}")
                except Exception as e:
                    logger.warning(f"讀取錯誤 {p}: {e}")

            if not images:
                continue



            # 新增流程：對每張圖進行 ROI 分割，展開為多個小圖
            processed_images = []
            processed_metadatas = []
            processed_ids = []

            for img, p in zip(images, valid_paths):
                try:
                    # 分割元件
                    components = extract_rois_from_image(img, top_n=5)
                    
                    if not components:
                        # 若分割失敗或無元件，退化為使用原圖 (視為單一元件)
                        components = [(img, {"component_index": 0, "bbox": [0,0,img.shape[1],img.shape[0]], "area": img.shape[0]*img.shape[1]})]

                    for roi_img, info in components:
                        processed_images.append(roi_img)
                        
                        # 構造 Metadata
                        filename = p.name
                        parent_pdf_id = p.stem
                        subdir_name = p.parent.name
                        comp_idx = info['component_index']
                        
                        meta = {
                            "original_filename": filename,
                            "page_num": 1,
                            "component_type": f"roi_{comp_idx}", # 標記第幾號元件
                            "parent_pdf_id": parent_pdf_id,
                            "category": subdir_name,
                            "path": str(p.absolute()),
                            "bbox": str([int(v) for v in info['bbox']]), # 轉為原生 int list 再轉字串
                            "area": int(info['area']) # 轉為原生 int
                        }
                        processed_metadatas.append(meta)
                        
                        # ID: parent_id + component_index
                        # 為避免重複執行造成 ID 衝突，可加上 hash 或時間戳，這裡保持簡單
                        unique_id = f"{parent_pdf_id}_roi_{comp_idx}"
                        processed_ids.append(unique_id)

                except Exception as e:
                    logger.error(f"影像處理失敗 (Segmentation) {p}: {e}")
                    continue

            if not processed_images:
                continue

            # 2. 提取特徵 (對分割後的所有 ROI 進行批次提取)
            # 由於 ROI 數量可能變多 (原 Batch Size * 5)，這裡可能需要再做一次 mini-batch
            # 但為簡化，直接送入 extract_batch (內部只是 forward pass)
            try:
                embeddings = self.extractor.extract_batch(processed_images)
            except Exception as e:
                logger.error(f"批次特徵提取失敗: {e}")
                continue

            if len(embeddings) == 0:
                continue

            # 3. 寫入資料庫
            try:
                self.db.upsert_vectors(
                    vectors=embeddings.tolist(),
                    metadatas=processed_metadatas,
                    ids=processed_ids
                )
                success_count += len(processed_ids) # 計算的是 indexed components 數量
            except Exception as e:
                logger.error(f"寫入資料庫失敗: {e}")

        logger.info(f"索引建立完成。成功索引: {success_count}/{total_images}")
        return success_count


