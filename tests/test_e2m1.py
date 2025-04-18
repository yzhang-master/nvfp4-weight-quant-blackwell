import pytest
import torch

from nvfp4 import E2M1_VALUES, decode_e2m1, encode_e2m1


def test_every_e2m1_bit_pattern_decodes_exactly():
    codes = torch.arange(16, dtype=torch.uint8)
    expected = torch.cat((E2M1_VALUES, -E2M1_VALUES))
    torch.testing.assert_close(decode_e2m1(codes), expected)
    assert torch.equal(encode_e2m1(expected), codes)


def test_rounding_is_nearest_with_ties_to_even_and_saturates():
    source = torch.tensor([0.24, 0.25, 0.26, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 99.0])
    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0])
    torch.testing.assert_close(decode_e2m1(encode_e2m1(source)), expected)
    torch.testing.assert_close(decode_e2m1(encode_e2m1(-source)), -expected)


def test_rejects_invalid_codes_and_values():
    with pytest.raises(ValueError, match="four bits"):
        decode_e2m1(torch.tensor([16], dtype=torch.uint8))
    with pytest.raises(ValueError, match="NaN"):
        encode_e2m1(torch.tensor([float("nan")]))
