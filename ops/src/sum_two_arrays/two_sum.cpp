#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void two_sum_launcher(const float* a, const float* b, float* c, int n);


void two_sum_gpu(at::Tensor a_tensor, at::Tensor b_tensor, at::Tensor c_tensor){
    CHECK_INPUT(a_tensor);
    CHECK_INPUT(b_tensor);
    CHECK_INPUT(c_tensor);

    const float* a = a_tensor.data_ptr<float>();
    const float* b = b_tensor.data_ptr<float>();
    float* c = c_tensor.data_ptr<float>();
    int n = a_tensor.size(0);
    two_sum_launcher(a, b, c, n);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &two_sum_gpu, "sum two arrays (CUDA)");
}