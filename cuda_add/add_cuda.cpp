#include <torch/torch.h>
#include <torch/extension.h>

// 声明CUDA接口函数
extern "C" void add_cuda_kernel(const float* a, const float* b, float* out, int n);

// 声明在CUDA文件中实现的函数
torch::Tensor add_cuda_forward(torch::Tensor x, torch::Tensor y);

at::Tensor add_cuda(at::Tensor a, at::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "Tensor a must be on CUDA device");
    TORCH_CHECK(b.device().is_cuda(), "Tensor b must be on CUDA device");
    TORCH_CHECK(a.sizes() == b.sizes(), "Tensor sizes must match");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Tensor b must be float32");

    at::Tensor out = torch::empty_like(a);
    int n = a.numel();

    add_cuda_kernel(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

// Python接口函数
torch::Tensor add_forward(torch::Tensor x, torch::Tensor y) {
    // 检查输入是否在CUDA设备上
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.device().is_cuda(), "y must be a CUDA tensor");
    
    return add_cuda_forward(x, y);
}

// 注册算子
TORCH_LIBRARY(my_ops, m) {
    m.def("add_cuda", add_cuda);
}

// 注册函数到Python模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_forward, "Custom add operation (CUDA)");
}