import torch

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    if not values.is_floating_point():
        raise TypeError("E2M1 encoding expects floating-point values")
    source = values.float()
    if not torch.isfinite(source).all():
        raise ValueError("E2M1 encoding rejects NaN and infinity")
    magnitude = source.abs().clamp(max=6.0)
    midpoints = E2M1_MIDPOINTS.to(source.device)
    codes = torch.bucketize(magnitude, midpoints, right=True)
    tied_odd_code = (codes % 2 == 1) & torch.isin(magnitude, midpoints)
    codes = codes - tied_odd_code.to(codes.dtype)
    sign = torch.signbit(source).to(torch.uint8) << 3
    return sign | codes.to(torch.uint8)


def decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8:
        raise TypeError("E2M1 codes must be uint8")
    if torch.any(codes > 15):
        raise ValueError("an E2M1 code must fit in four bits")
    table = torch.cat((E2M1_VALUES, -E2M1_VALUES)).to(codes.device)
    return table[codes.long()]
