#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void sum_reduce_launcher(const float* array, float* ans, int n);


void sum_reduce_gpu(at::Tensor array_tensor, at::Tensor ans_tensor){
    CHECK_INPUT(array_tensor);
    CHECK_CUDA(ans_tensor);

    const float* array = array_tensor.data_ptr<float>();
    float* ans = ans_tensor.data_ptr<float>();
    int n = array_tensor.size(0);
    sum_reduce_launcher(array, ans, n);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sum_reduce_gpu, "sum an array (CUDA)");
}