import torch
import triton
import triton.language as tl


@triton.jit
def kernel_rgb_to_grey(
    in_ptr,
    out_ptr,
    height,
    width,
    bs_x: tl.constexpr,
    bs_y: tl.constexpr,
):
    """Triton kernel for converting an RGB image to its greyscale counterpart."""
    # Get horizontal program id and vertical program id
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    # Horizontal and vertical offsets
    # This is the rows and columns that define which rectangular block of the data to work on
    offsets_x = pid_x * bs_x + tl.arange(0, bs_x)
    offsets_y = pid_y * bs_y + tl.arange(0, bs_y)

    # Combine offsets.
    # Broadcasting of shapes: (bs_x, 1) + (1, bs_y) -> (bs_x, bs_y)
    # Row-major memory layout is assumed!
    offsets = width * tl.expand_dims(offsets_x, axis=1) + tl.expand_dims(
        offsets_y, axis=0
    )

    # Horizontal and vertical masks
    mask_x = offsets_x < height
    mask_y = offsets_y < width

    # Combine masks: the data must not go out-of-bounds horizontally or vertically,
    # hence we need & (logical 'and')
    # Broadcasting of shapes: (bs_x, 1) & (1, bs_y) -> (bs_x, bs_y)
    # For example:
    #        6  7
    #       13 14
    #
    #       True   &   True False   =   True True   &   True False   =  True False
    #       True                        True True       True False      True False
    mask = tl.expand_dims(mask_x, axis=1) & tl.expand_dims(mask_y, axis=0)

    # Load data
    r = tl.load(in_ptr + 0 * height * width + offsets, mask=mask)
    g = tl.load(in_ptr + 1 * height * width + offsets, mask=mask)
    b = tl.load(in_ptr + 2 * height * width + offsets, mask=mask)

    # Convert from RGB to grey
    out = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)


def rgb_to_grey(x, block_size):
    C, H, W = x.shape

    # Allocate space for output
    out = torch.empty((H, W), dtype=x.dtype, device=x.device)

    # Define grid
    grid = lambda meta: (triton.cdiv(H, meta["bs_x"]), triton.cdiv(W, meta["bs_y"]))

    # Launch kernel
    kernel_rgb_to_grey[grid](x, out, H, W, bs_x=block_size[0], bs_y=block_size[1])

    return out.view(H, W)
