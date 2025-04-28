# nvfp4-weight-quant-blackwell

> **Era:** 2025+ · Staff AI Engineer focus area (narrow-precision LLM inference). Technique from 2025, reimplemented in 2026 with CUTLASS 4.x and PyTorch 2.x.

A CPU-exact NVFP4 quantizer and TN GEMM reference, plus a real CUTLASS block-scaled SM100 executable. Values are signed E2M1, scales are positive FP8 E4M3, and every 16 values along K share one scale.

## Problem

Four-bit weight storage reduces memory traffic, but a fake low-bit format can hide accuracy and packing errors. NVFP4 also requires two distinct low-precision decisions: rounding each value to E2M1 and rounding each block scale to E4M3. Blackwell consumes packed operands and a tiled scale-factor layout, not an ordinary row-major scale matrix.

## Method

For each 16-value block `b` along K:

```
s = fp8_e4m3(max(abs(b)) / 6)     q = e2m1_nearest_even(clamp(b / s, -6, 6))
```

E2M1's complete magnitude set is `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. The implementation enumerates every bit pattern and uses nearest rounding with ties to even. Two codes occupy each byte, low nibble first. Zero blocks use scale 1. Dequantization multiplies decoded values by the **rounded** FP8 scale, so the test path includes scale quantization error rather than keeping a hidden fp32 scale.

The reference TN GEMM takes A as `M×K` and B as `N×K` and computes, in fp32:

```
C[m,n] = sum_kb sa[m,kb] * sb[n,kb] *
         sum_r decode(Qa[m,kb,r]) * decode(Qb[n,kb,r])
```

The CUDA executable uses `cutlass::nv_float4_t<cutlass::float_e2m1_t>`, a `128×128×256` MMA tile and CUTLASS's `Sm1xxBlkScaledConfig` scale-factor layout. Its hardware scale type is unsigned `float_ue4m3_t`; positive values have the same E4M3 magnitude encoding as the CPU representation. The executable verifies against CUTLASS's block-scaled host GETT reference before timing and reports GEMM tokens/s as `M / latency`, not end-to-end model throughput.

## Results

Measured locally on CPU, seed 123, `M=32, N=32, K=64`, with K-dependent weight dynamic range:

| Metric | NVFP4 | FP8 E4M3, same 16-value blocks |
|---|---:|---:|
| GEMM normalized RMSE vs fp32 | 0.1314934939 | 0.0315982401 |
| weight normalized RMSE | 0.0974440277 | 0.0216094125 |
| storage for both operands | 2304 bytes | 5120 bytes |

BF16 storage for the same operands is 8192 bytes; measured NVFP4/BF16 storage ratio is **0.28125**, including scales and padding. FP8 uses fp32 scales in this comparison, while NVFP4 uses its required FP8 scales. These are accuracy/storage measurements, not runtime results.

SM100 compilation, correctness, TFLOPS and GEMM tokens/s are **pending** because this machine has no CUDA toolkit or GPU:

| GPU | shape M×N×K | NVFP4 TFLOPS | GEMM tokens/s |
|---|---|---:|---:|
| _pending B200_ | 128×1024×1024 | pending | pending |

Produce those measurements with:

```bash
CUTLASS_DIR=/path/to/cutlass scripts/bench_sm100.sh
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
pytest -q
python -m nvfp4.bench --smoke
```

For datacenter Blackwell SM100/SM103, install CUDA 12.8+, CMake 3.24+ and CUTLASS 4.x, then:

```bash
CUTLASS_DIR=/path/to/cutlass scripts/build_sm100.sh
build/sm100/nvfp4_sm100 --m=128 --n=1024 --k=1024 --warmup=10 --iterations=50
# Add --no-verify only for timing after a verified run.
```

No script manufactures a B200 measurement on a CPU.

## Limitations

- This implements NVFP4 value quantization and local FP8 scales; it does not implement a model-specific global calibration scale or Hadamard preprocessing.
- Extreme blocks above the E4M3 scale range saturate. Production weight conversion should calibrate or reject such tensors.
- Python packing is linear low-nibble-first storage. CUTLASS's interleaved scale-factor layout is constructed by the C++ executable; the Python packer does not claim to serialize that layout.
- The CUDA path requires TN operands and K divisible by 256. It generates its own packed test inputs and measures a single GEMM, not a complete LLM decode pipeline.
- The SM100 source has not been compiled or run locally; its correctness and performance remain pending.

## References

- NVIDIA. *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.* 2025. https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- NVIDIA CUTLASS. *Blackwell SM100 GEMMs.* https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- NVIDIA CUTLASS, example 72a: Blackwell NVFP4 BF16 GEMM. https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu

## License

MIT
