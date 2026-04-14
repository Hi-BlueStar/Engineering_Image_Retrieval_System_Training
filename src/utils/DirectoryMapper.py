import json
import os
from pathlib import Path


class DirectoryMapper:
    """
    專業級資料夾結構映射器。
    支援輸出為視覺化樹狀圖 (.txt) 或 結構化資料 (.json)。
    """

    def __init__(self, root_path: str, ignore_patterns: list[str] | None = None):
        self.root_path = Path(root_path)
        # 預設忽略的目錄，保持輸出整潔
        self.ignore_patterns = ignore_patterns or [
            ".git",
            "__pycache__",
            ".idea",
            ".vscode",
            "node_modules",
        ]

        if not self.root_path.exists():
            raise FileNotFoundError(f"找不到路徑: {self.root_path}")

    def _should_ignore(self, path: Path) -> bool:
        """判斷是否應該忽略該路徑"""
        return any(pattern in path.parts for pattern in self.ignore_patterns)

    def generate_tree_text(self, current_path: Path = None, prefix: str = "") -> str:
        """
        生成類似 Linux 'tree' 指令的文字結構。
        """
        if current_path is None:
            current_path = self.root_path
            return f"{current_path.name}/\n" + self.generate_tree_text(current_path, "")

        output = ""
        try:
            # 取得並排序內容，讓資料夾與檔案排列整齊
            items = sorted(
                list(current_path.iterdir()),
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )

            # 過濾掉不需要的項目
            items = [item for item in items if not self._should_ignore(item)]

            count = len(items)
            for i, item in enumerate(items):
                is_last = i == count - 1
                connector = "└── " if is_last else "├── "

                output += (
                    f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}\n"
                )

                if item.is_dir():
                    extension = "    " if is_last else "│   "
                    output += self.generate_tree_text(item, prefix + extension)

        except PermissionError:
            output += f"{prefix}└── [存取被拒]\n"

        return output

    def generate_json_structure(self, current_path: Path = None) -> dict:
        """
        遞迴生成 JSON 結構化字典，包含檔案大小等元數據 (Metadata)。
        """
        if current_path is None:
            current_path = self.root_path

        node = {
            "name": current_path.name,
            "path": str(current_path),
            "type": "directory" if current_path.is_dir() else "file",
        }

        if current_path.is_file():
            try:
                node["size_bytes"] = current_path.stat().st_size
            except Exception:
                node["size_bytes"] = -1

        if current_path.is_dir():
            children = []
            try:
                items = sorted(
                    list(current_path.iterdir()), key=lambda x: x.name.lower()
                )
                for item in items:
                    if not self._should_ignore(item):
                        children.append(self.generate_json_structure(item))
            except PermissionError:
                children.append({"name": "[ACCESS DENIED]", "type": "error"})

            node["children"] = children

        return node

    def save_to_file(
        self, output_format: str = "txt", output_filename: str = "structure_output"
    ):
        """
        將結果寫入檔案的統一接口。
        """
        full_filename = f"{output_filename}.{output_format}"

        print(f"正在生成 {output_format.upper()} 結構...")

        try:
            if output_format == "txt":
                content = self.generate_tree_text()
                with open(full_filename, "w", encoding="utf-8") as f:
                    f.write(content)

            elif output_format == "json":
                content = self.generate_json_structure()
                with open(full_filename, "w", encoding="utf-8") as f:
                    json.dump(content, f, indent=4, ensure_ascii=False)

            else:
                raise ValueError("不支援的格式。僅支援 'txt' 或 'json'。")

            print(f"✅ 成功儲存至: {os.path.abspath(full_filename)}")

        except Exception as e:
            print(f"❌ 儲存失敗: {str(e)}")


# --- 使用範例 ---
if __name__ == "__main__":
    # 設定要掃描的路徑 (預設為當前目錄)
    target_directory = "dataset"

    mapper = DirectoryMapper(target_directory)

    # 1. 輸出為 TXT (視覺化樹狀圖)
    mapper.save_to_file(output_format="txt", output_filename="folder_structure")

    # 2. 輸出為 JSON (資料交換格式)
    mapper.save_to_file(output_format="json", output_filename="folder_structure")
