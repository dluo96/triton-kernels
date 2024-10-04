import torch

from tk.matmul import kernel_matmul_naive, matmul


class MatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # Use the context object to store information needed
        # during the backward pass
        ctx.save_for_backward(a, b)
        output = matmul(a, b, kernel_matmul=kernel_matmul_naive)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        c = a @ b
        dLda = dLdc @ b^T
        dLdb = a^T @ dLdc
        """
        a, b = ctx.saved_tensors
        dcda = MatMul.apply(grad_output, b.T)
        dcdb = MatMul.apply(a.T, grad_output)

        return dcda, dcdb
