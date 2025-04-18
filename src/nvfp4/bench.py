import argparse
import json
import platform
from pathlib import Path

import torch

from .gemm import nvfp4_gemm
from .quantization import (
    dequantize_fp8_blocks,
    quantize_fp8_blocks,
    quantize_nvfp4,
)


def _nrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual - expected).square().mean().sqrt()
    return float(error / expected.square().mean().sqrt())


def measure_smoke(seed: int = 123) -> dict[str, float | int | str]:
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(32, 64, generator=generator)
    b = torch.randn(32, 64, generator=generator) * torch.logspace(-1, 1, 64)[None, :]
    qa = quantize_nvfp4(a)
    qb = quantize_nvfp4(b)
    actual = nvfp4_gemm(qa, qb)
    expected = a @ b.T
    fp8_a_values, fp8_a_scales = quantize_fp8_blocks(a)
    fp8_b_values, fp8_b_scales = quantize_fp8_blocks(b)
    fp8_a = dequantize_fp8_blocks(fp8_a_values, fp8_a_scales, a.shape[1])
    fp8_b = dequantize_fp8_blocks(fp8_b_values, fp8_b_scales, b.shape[1])
    nvfp4_bytes = qa.storage_bytes + qb.storage_bytes
    fp8_bytes = (
        fp8_a_values.numel()
        + fp8_b_values.numel()
        + 4 * (fp8_a_scales.numel() + fp8_b_scales.numel())
    )
    bf16_bytes = 2 * (a.numel() + b.numel())
    return {
        "device": "cpu",
        "seed": seed,
        "m": a.shape[0],
        "n": b.shape[0],
        "k": a.shape[1],
        "nvfp4_gemm_nrmse": _nrmse(actual, expected),
        "fp8_gemm_nrmse": _nrmse(fp8_a @ fp8_b.T, expected),
        "nvfp4_weight_nrmse": _nrmse(qb.dequantize(), b),
        "fp8_weight_nrmse": _nrmse(fp8_b, b),
        "nvfp4_storage_bytes": nvfp4_bytes,
        "fp8_storage_bytes": fp8_bytes,
        "bf16_storage_bytes": bf16_bytes,
        "nvfp4_vs_bf16_memory_ratio": nvfp4_bytes / bf16_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measured CPU accuracy/storage smoke")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.smoke:
        parser.error("use --smoke; SM100 performance is produced by scripts/bench_sm100.sh")
    result = measure_smoke()
    result["torch"] = torch.__version__
    result["python"] = platform.python_version()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
