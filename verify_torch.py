import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# 測試一個簡單的張量運算
x = torch.rand(5, 3)
print("\nRandom Tensor:")
print(x)
