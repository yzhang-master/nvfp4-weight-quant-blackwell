import pytest
import torch

from nvfp4 import PackedNVFP4, nvfp4_gemm, quantize_nvfp4


def _normalized_rmse(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    return (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()


def test_gemm_matches_dequantized_product_for_partial_shapes():
    generator = torch.Generator().manual_seed(41)
    a = torch.randn(33, 67, generator=generator)
    b = torch.randn(35, 67, generator=generator)
    qa = quantize_nvfp4(a)
    qb = quantize_nvfp4(b)
    actual = nvfp4_gemm(qa, qb)
    expected = qa.dequantize() @ qb.dequantize().T
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert _normalized_rmse(actual, a @ b.T) < 0.2


def test_each_k_block_scale_contributes_to_accumulation():
    a = quantize_nvfp4(torch.full((16, 32), 6.0))
    b = quantize_nvfp4(torch.full((16, 32), 6.0))
    expected = nvfp4_gemm(a, b)
    scales = a.scales.float().clone()
    scales[:, 1] *= 3
    mutated = PackedNVFP4(
        a.packed, scales.to(torch.float8_e4m3fn), a.original_shape
    )
    torch.testing.assert_close(nvfp4_gemm(mutated, b), expected * 2)


def test_gemm_checks_shapes():
    a = quantize_nvfp4(torch.ones(2, 16))
    b = quantize_nvfp4(torch.ones(17, 2))
    with pytest.raises(ValueError, match="incompatible TN GEMM"):
        nvfp4_gemm(a, b)
