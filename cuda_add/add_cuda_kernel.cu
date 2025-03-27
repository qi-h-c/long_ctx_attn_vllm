#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>  // 用于主机代码中的printf

// 在设备代码中不使用printf
template <typename scalar_t>
__global__ void add_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ output,
    const int size) {
    
    // 计算当前线程的全局索引
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保不会超出数组边界
    if (index < size) {
        output[index] = x[index] + y[index];
    }
}

// 在主机代码中使用printf检查CUDA错误
torch::Tensor add_cuda_forward(torch::Tensor x, torch::Tensor y) {
    // 确保输入在同一设备上
    TORCH_CHECK(x.device() == y.device(), "x and y must be on the same device");
    // 确保形状相同
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    
    auto output = torch::empty_like(x);
    
    // 计算元素总数
    const int num_elements = x.numel();
    
    // 设置CUDA网格和块
    const int threads = 1024;
    const int blocks = (num_elements + threads - 1) / threads;
    
    // 根据输入数据类型分派不同的内核
    AT_DISPATCH_FLOATING_TYPES(x.type(), "add_cuda_forward", ([&] {
        add_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements
        );
        
        // 使用TORCH_CHECK或C++的标准输出来检查错误，避免直接使用printf
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    }));
    
    return output;
}


// 包装为C接口函数，这是add_cuda.cpp中使用extern "C"声明的函数
extern "C" void add_cuda_kernel(const float* a, const float* b, float* out, int n) {
    // 确定网格和块的大小
    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;
    
    // 调用kernel
    add_kernel<float><<<blocks, threads>>>(a, b, out, n);
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 使用CUDA API检查错误
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }
}