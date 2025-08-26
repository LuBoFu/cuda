# compare_cpu_gpu.py
import torch
import time
import numpy as np

def benchmark_matmul():
    # 矩阵大小
    N = 8192
    dtype = torch.float32

    print(f"Matrix size: {N}x{N}, dtype: {dtype}")
    print("Generating random matrices...")
    
    # ====================
    # CPU 测试
    # ====================
    print("Running CPU matrix multiplication...")
    a_cpu = torch.randn(N, N, dtype=dtype)
    b_cpu = torch.randn(N, N, dtype=dtype)
    
    # 预热
    for _ in range(3):
        torch.matmul(a_cpu, b_cpu)
    
    # 正式计时（多次运行取平均）
    start_time = time.time()
    for _ in range(10):
        c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = (time.time() - start_time) / 10
    print(f"CPU Time: {cpu_time:.4f} seconds")

    # ====================
    # GPU 测试
    # ====================
    if not torch.cuda.is_available():
        print("GPU not available. Skipping GPU test.")
        return

    device = torch.device('cuda')
    print("Running GPU matrix multiplication...")

    # 数据提前传到 GPU
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)

    # 同步 + 预热
    torch.cuda.synchronize()
    for _ in range(5):
        torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()

    # 使用 CUDA Events 精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(10):
        torch.matmul(a_gpu, b_gpu)
    end_event.record()
    torch.cuda.synchronize()  # 等待完成

    gpu_time = start_event.elapsed_time(end_event) / 10 / 1000  # 转为秒
    print(f"GPU Time: {gpu_time:.4f} seconds")

    # ====================
    # 加速比 & 验证
    # ====================
    speedup = cpu_time / gpu_time
    print(f"Speedup (CPU/GPU): {speedup:.2f}x")

    # 将 GPU 结果取回 CPU 进行验证
    with torch.no_grad():
        c_gpu = torch.matmul(a_gpu, b_gpu).cpu()

    # 计算误差
    max_error = torch.max(torch.abs(c_cpu - c_gpu)).item()
    mean_error = torch.mean(torch.abs(c_cpu - c_gpu)).item()

    print(f"Max error: {max_error:.6e}")
    print(f"Mean error: {mean_error:.6e}")
    
    tolerance = 1e-3
    if max_error < tolerance:
        print("✅ GPU result is correct (within tolerance).")
    else:
        print("❌ GPU result has significant error!")

    # 输出硬件信息
    print("\nHardware Info:")
    print(f"CPU: {torch.get_num_threads()} threads")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
    benchmark_matmul()
