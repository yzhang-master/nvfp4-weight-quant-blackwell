import torch

from nvfp4 import PackedNVFP4, decode_e2m1, encode_e2m1, quantize_nvfp4


def test_nibbles_pack_low_value_first_and_pad_edges():
    codes = torch.arange(17, dtype=torch.float32) % 8
    source = decode_e2m1(codes.to(torch.uint8)).view(1, 17)
    quantized = quantize_nvfp4(source)
    unpacked = quantized.unpack()
    assert quantized.packed.shape == (1, 16)
    assert torch.equal(unpacked[0, :17], codes.to(torch.uint8))
    assert torch.count_nonzero(unpacked[0, 17:]) == 0
    expected_byte = int(codes[0]) | (int(codes[1]) << 4)
    assert quantized.packed[0, 0] == expected_byte


def test_scales_are_fp8_and_applied_per_16_values():
    source = torch.zeros(2, 32)
    source[0, :16] = 6
    source[0, 16:] = 0.75
    source[1] = 0
    quantized = quantize_nvfp4(source)
    assert quantized.scales.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(quantized.scales.float(), torch.tensor([[1.0, 0.125], [1.0, 1.0]]))
    torch.testing.assert_close(quantized.dequantize(), source)


def test_quantization_is_close_on_dynamic_ranges_and_partial_blocks():
    generator = torch.Generator().manual_seed(37)
    source = torch.randn(33, 67, generator=generator) * torch.logspace(-2, 1, 67)
    restored = quantize_nvfp4(source).dequantize()
    assert restored.shape == source.shape
    relative_rmse = (restored - source).square().mean().sqrt() / source.square().mean().sqrt()
    assert relative_rmse < 0.16


def test_mutated_scale_changes_reconstruction():
    quantized = quantize_nvfp4(torch.full((1, 16), 3.0))
    mutated = PackedNVFP4(
        quantized.packed,
        (quantized.scales.float() * 2).to(torch.float8_e4m3fn),
        quantized.original_shape,
    )
    torch.testing.assert_close(mutated.dequantize(), quantized.dequantize() * 2)


def test_signed_zero_preserves_its_encoding_sign():
    codes = encode_e2m1(torch.tensor([-0.0, 0.0]))
    assert torch.equal(codes, torch.tensor([8, 0], dtype=torch.uint8))
