import torch

from .e2m1 import decode_e2m1
from .quantization import BLOCK_SIZE, PackedNVFP4


def nvfp4_gemm(a: PackedNVFP4, b: PackedNVFP4) -> torch.Tensor:
    if a.original_shape[1] != b.original_shape[1]:
        raise ValueError(
            f"incompatible TN GEMM shapes: {a.original_shape} and {b.original_shape}"
        )
    if a.packed.device != b.packed.device or a.scales.device != b.scales.device:
        raise ValueError("both operands must be on the same device")
    if a.padded_shape[1] != b.padded_shape[1]:
        raise ValueError("operands have incompatible padded K dimensions")

    a_values = decode_e2m1(a.unpack()).view(a.padded_shape[0], -1, BLOCK_SIZE)
    b_values = decode_e2m1(b.unpack()).view(b.padded_shape[0], -1, BLOCK_SIZE)
    output = torch.zeros(
        (a.padded_shape[0], b.padded_shape[0]), dtype=torch.float32, device=a.packed.device
    )
    for block_k in range(a.padded_shape[1] // BLOCK_SIZE):
        a_block = a_values[:, block_k, :]
        b_block = b_values[:, block_k, :]
        scale = a.scales.float()[:, block_k, None] * b.scales.float()[None, :, block_k]
        output.add_((a_block @ b_block.T) * scale)
    return output[: a.original_shape[0], : b.original_shape[0]]
