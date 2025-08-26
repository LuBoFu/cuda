import numpy as np
from numba import cuda, jit, float32
import time

# 设置矩阵大小（可调整测试规模）
N = 8192
# 数据类型（使用 float32 更适合 GPU）
dtype = np.float32

print(f"Matrix size: {N}x{N}, dtype: {dtype.__name__}")

# ================================
# 1. CPU 版本：使用 NumPy
# ================================
def cpu_matrix_multiply(A, B):
    return np.dot(A, B)

# ================================
# 2. GPU 版本：使用 Numba CUDA
# ================================
@cuda.jit
def gpu_matrix_multiply_kernel(C, A, B):
    i = cuda.grid(1)
    if i < C.shape[0]:
        sum = 0.0
        for k in range(A.shape[1]):
            sum += A[i, k] * B[k, i % B.shape[1]]
        C[i, i % B.shape[1]] = sum

# 更高效的二维网格版本
@cuda.jit('void(float32[:,::1], float32[:,::1], float32[:,::1])')
def matmul_kernel_2d(C, A, B):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        temp = 0.0
        for k in range(A.shape[1]):
            temp += A[row, k] * B[k, col]
        C[row, col] = temp

def gpu_matrix_multiply(A, B):
    # 将数据复制到设备
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array((A.shape[0], B.shape[1]), dtype=dtype)

    # 配置线程网格
    threads_per_block = (16, 16)  # 每个 block 16x16 线程
    blocks_per_grid_x = (A.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (B.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # 执行内核
    matmul_kernel_2d[blocks_per_grid, threads_per_block](d_C, d_A, d_B)

    # 复制结果回主机
    C = d_C.copy_to_host()
    return C

# ================================
# 主程序：性能对比
# ================================
def main():
    # 生成随机矩阵
    print("Generating random matrices...")
    A = np.random.rand(N, N).astype(dtype)
    B = np.random.rand(N, N).astype(dtype)

    # CPU 计算
    print("Running CPU matrix multiplication...")
    start = time.time()
    C_cpu = cpu_matrix_multiply(A, B)
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.4f} seconds")

    # GPU 计算
    print("Running GPU matrix multiplication...")
    start = time.time()
    C_gpu = gpu_matrix_multiply(A, B)
    gpu_time = time.time() - start
    print(f"GPU Time: {gpu_time:.4f} seconds")

    # 性能对比
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    print(f"Speedup (CPU/GPU): {speedup:.2f}x")

    # 验证结果（取一小块比较）
    print("Validating results...")
    diff = np.abs(C_cpu - C_gpu)
    max_error = np.max(diff)
    mean_error = np.mean(diff)
    print(f"Max error: {max_error:.6e}")
    print(f"Mean error: {mean_error:.6e}")

    if max_error < 1e-2:
        print("✅ GPU result is correct (within tolerance).")
    else:
        print("❌ GPU result differs significantly!")

if __name__ == "__main__":
    main()
