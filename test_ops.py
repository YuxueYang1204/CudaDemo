from ops import sum_single_op, sum_double_op
import torch
import time

class Timer:
    def __init__(self, op_name):
        self.begin_time = 0
        self.end_time = 0
        self.op_name = op_name

    def __enter__(self):
        torch.cuda.synchronize()
        self.begin_time = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        print(f"Average time cost of {self.op_name} is {(self.end_time - self.begin_time) * 1000} ms")


if __name__ == '__main__':
    n = 10000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    array1 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=True)
    array2 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=True)
    with Timer("sum_single"):
        ans = sum_single_op(array1)
    assert ans == n
    with Timer("sum_double"):
        ans = sum_double_op(array1, array2)
    assert (ans == 2).all()