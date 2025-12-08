import torch
import torchvision

def check_system_info():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("警告: 系統未偵測到 CUDA 裝置，將使用 CPU。")

if __name__ == "__main__":
    check_system_info()