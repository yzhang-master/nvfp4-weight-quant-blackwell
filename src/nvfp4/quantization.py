from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .e2m1 import decode_e2m1, encode_e2m1

BLOCK_SIZE = 16
E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0


@dataclass(frozen=True)
class PackedNVFP4:
    packed: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]

    @property
    def padded_shape(self) -> tuple[int, int]:
        return self.original_shape[0], self.packed.shape[1] * 2

    def unpack(self) -> torch.Tensor:
        low = self.packed & 0x0F
        high = self.packed >> 4
        return torch.stack((low, high), dim=-1).reshape(self.padded_shape)

    def dequantize(self) -> torch.Tensor:
        codes = self.unpack()
        blocks = codes.view(self.padded_shape[0], -1, BLOCK_SIZE)
        values = decode_e2m1(blocks) * self.scales.float().unsqueeze(-1)
        output = values.reshape(self.padded_shape)
        return output[:, : self.original_shape[1]]

    @property
    def storage_bytes(self) -> int:
        return self.packed.numel() + self.scales.numel()


def quantize_nvfp4(tensor: torch.Tensor) -> PackedNVFP4:
    if tensor.ndim != 2:
        raise ValueError("NVFP4 quantization expects a rank-2 tensor")
    if not tensor.is_floating_point():
        raise TypeError("NVFP4 quantization expects a floating-point tensor")
    if not torch.isfinite(tensor).all():
        raise ValueError("input contains NaN or infinity")
    source = tensor.float()
    rows, cols = source.shape
    padded_cols = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    padded = F.pad(source, (0, padded_cols - cols))
    blocks = padded.view(rows, padded_cols // BLOCK_SIZE, BLOCK_SIZE)
    absmax = blocks.abs().amax(dim=-1)
    raw_scales = torch.where(absmax == 0, torch.ones_like(absmax), absmax / E2M1_MAX)
    scales = raw_scales.clamp(max=FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    normalized = blocks / scales.float().unsqueeze(-1)
    codes = encode_e2m1(normalized)
    flat_codes = codes.reshape(rows, padded_cols)
    packed = flat_codes[:, 0::2] | (flat_codes[:, 1::2] << 4)
    return PackedNVFP4(packed=packed, scales=scales, original_shape=(rows, cols))


def quantize_fp8_blocks(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 comparison with the same 16-value blocks and an fp32 scale."""
    source = tensor.float()
    rows, cols = source.shape
    padded_cols = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    padded = F.pad(source, (0, padded_cols - cols))
    blocks = padded.view(rows, -1, BLOCK_SIZE)
    absmax = blocks.abs().amax(dim=-1)
    scales = torch.where(absmax == 0, torch.ones_like(absmax), absmax / FP8_E4M3_MAX)
    values = (blocks / scales.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return values.to(torch.float8_e4m3fn), scales


def dequantize_fp8_blocks(
    values: torch.Tensor, scales: torch.Tensor, original_cols: int
) -> torch.Tensor:
    restored = values.float() * scales.unsqueeze(-1)
    return restored.reshape(values.shape[0], -1)[:, :original_cols]
