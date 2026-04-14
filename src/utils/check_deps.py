import os
import ast
import sys
import stdlib_list

def get_imports(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            root = ast.parse(f.read(), path)
        except Exception as e:
            # 略過語法錯誤的檔案
            return set()

    imports = set()
    for node in ast.walk(root):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def main():
    project_path = '.'  # 當前目錄
    all_imports = set()
    
    # 獲取對應 Python 版本的標準庫列表
    try:
        # 需先 pip install stdlib-list
        std_libs = set(stdlib_list.stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}"))
    except ImportError:
        print("建議安裝 stdlib-list 以獲得精確過濾: pip install stdlib-list")
        std_libs = set(sys.builtin_module_names)

    print(f"正在分析 {os.path.abspath(project_path)} ...")

    for root, dirs, files in os.walk(project_path):
        # 排除虛擬環境目錄 (常見名稱)
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', 'env', '.git', '__pycache__']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                all_imports.update(get_imports(file_path))

    # 過濾標準庫和本地專案模組 (簡單判斷)
    # 這裡假設專案內的模組名稱不會與第三方庫重名
    third_party = {imp for imp in all_imports if imp not in std_libs}

    print("\n--- 偵測到的潛在第三方依賴 ---")
    for lib in sorted(third_party):
        print(lib)

if __name__ == "__main__":
    main()