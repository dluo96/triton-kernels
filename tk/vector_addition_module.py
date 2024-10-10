import torch

from tk.vector_addition import add_vectors


class AddVectors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # The context object can be used to store information that the backward pass
        # can use
        ctx.save_for_backward(x, y)

        # Call the Triton kernel to perform the forward pass (addition of vectors)
        output = add_vectors(x, y)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Retrieve saved tensors from forward pass
        x, y = ctx.saved_tensors

        # Compute the gradient with respect to each input.
        # The gradients for addition are straightforward; both x and y
        # have gradients equal to grad_output because ∂(x + y)/∂x = 1 and ∂(x + y)/∂y = 1
        grad_x = grad_output.clone()
        grad_y = grad_output.clone()

        # Return gradients for each input (x and y)
        return grad_x, grad_y
