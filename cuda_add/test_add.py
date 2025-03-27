import torch
import my_ops
import time

# 创建两个CUDA张量
a = torch.rand(1000000, device='cuda')
b = torch.rand(1000000, device='cuda')

# 确保CUDA初始化完成
torch.cuda.synchronize()

# 测试自定义加法
start = time.time()
c_custom = my_ops.add(a, b)
torch.cuda.synchronize()
custom_time = time.time() - start

# 测试PyTorch内置加法
start = time.time()
c_torch = a + b
torch.cuda.synchronize()
torch_time = time.time() - start

# 验证结果
print(f"结果是否一致: {torch.allclose(c_custom, c_torch)}")
print(f"自定义加法耗时: {custom_time*1000:.4f} ms")
print(f"PyTorch加法耗时: {torch_time*1000:.4f} ms")