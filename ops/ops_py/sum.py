import torch
from torch.autograd import Function
import sum_single
import sum_double

class SumSingle(Function):

    @staticmethod
    def forward(ctx, array):
        """sum_single function forward.
        Args:
            array (torch.Tensor): [n,]
        
        Returns:
            ans (torch.Tensor): [1,]
        """
        array = array.float()
        ans = array.new_zeros(1)
        sum_single.forward(array.contiguous(), ans)

        # ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation
        ctx.shape = array.shape[0]

        return ans

    @staticmethod
    def backward(ctx, g_out):
        # return None   # if the function is no need for backpropogation

        n = ctx.shape
        g_in = g_out.new_ones(n) * g_out
        return g_in


class SumDouble(Function):

    @staticmethod
    def forward(ctx, array1, array2):
        """sum_double function forward.
        Args:
            array1 (torch.Tensor): [n,]
            array2 (torch.Tensor): [n,]
        
        Returns:
            ans (torch.Tensor): [n,]
        """
        array1 = array1.float()
        array2 = array2.float()
        ans = array1.new_zeros(array1.shape)
        sum_double.forward(array1.contiguous(), array2.contiguous(), ans)

        # ctx.mark_non_differentiable(ans) # if the function is no need for backpropogation

        return ans

    @staticmethod
    def backward(ctx, g_out):
        # return None, None   # if the function is no need for backpropogation

        g_in1 = g_out.clone()
        g_in2 = g_out.clone()
        return g_in1, g_in2


sum_single_op = SumSingle.apply
sum_double_op = SumDouble.apply