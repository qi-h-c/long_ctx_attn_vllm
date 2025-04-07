#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>


extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}


namespace extension_cpp {

// CPU 实现：逐元素加法
at::Tensor add_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());  // 确保 a 和 b 的尺寸一致
  TORCH_CHECK(a.dtype() == at::kFloat);  // 确保 a 和 b 的数据类型为 float
  TORCH_CHECK(b.dtype() == at::kFloat);  // 同上
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);  // 确保 a 和 b 在 CPU 上
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);  // 同上

  // 将 a 和 b 转换为连续内存，确保能够高效访问
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  
  // 创建输出 tensor
  at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());

  const float* a_ptr = a_contig.data_ptr<float>();  // 获取 a 的数据指针
  const float* b_ptr = b_contig.data_ptr<float>();  // 获取 b 的数据指针
  float* result_ptr = result.data_ptr<float>();  // 获取输出 tensor 的数据指针

  // 获取元素数量
  int64_t numel = a_contig.numel();
  
  // 执行逐元素加法操作
  for (int64_t i = 0; i < numel; i++) {
    result_ptr[i] = a_ptr[i] + b_ptr[i];  // 执行 a + b
  }

  return result;  // 返回计算结果
}

// 定义操作
TORCH_LIBRARY(extension_cpp, m) {
  m.def("add(Tensor a, Tensor b) -> Tensor");  // 定义 add 操作
}

// 注册 CPU 实现
TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("add", &add_cpu);  // 注册 add 操作的 CPU 实现
}

}  // namespace extension_cpp
