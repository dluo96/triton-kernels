import torch

from tk.matmul import kernel_matmul_naive, matmul


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        output = matmul(a, b, kernel_matmul=kernel_matmul_naive)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        c = a @ b
        dLda = dLdc @ b^T
        """
        a, b = ctx.saved_tensors
        # grad_output is dL/dc
        # We compute dc/da and dc/db
        dcda = MatMul.apply(grad_output, b.T)
        dcdb = MatMul.apply(a.T, grad_output)

        return dcda, dcdb
