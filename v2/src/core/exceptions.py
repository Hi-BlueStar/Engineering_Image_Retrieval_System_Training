"""
專案自定義的錯誤型別 (Custom Exceptions)。

定義專案領域邏輯專屬的例外狀況，便於錯誤捕捉與隔離，
避免使用通用 Exception 導致除錯困難。
"""

class ConfigError(Exception):
    """配置相關錯誤，例如 YAML 解析失敗或必填欄位遺失。"""
    pass

class DataPipelineError(Exception):
    """資料管線錯誤，例如找不到來源資料或前處理過程崩潰。"""
    pass

class ModelInitializationError(Exception):
    """模型初始化過程中的錯誤。"""
    pass
