#include <iostream>
#include <stdexcept>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentAB = 32;
constexpr int AlignmentCD = 8;
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    ElementC, LayoutC, AlignmentCD, ElementD, LayoutD, AlignmentCD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentAB, ElementB, LayoutB, AlignmentAB, float,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

struct Options {
  int m = 128;
  int n = 1024;
  int k = 1024;
  int warmup = 10;
  int iterations = 50;
  bool verify = true;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);
    if (cmd.check_cmd_line_flag("no-verify")) verify = false;
    if (m <= 0 || n <= 0 || k <= 0 || k % 256) {
      throw std::invalid_argument("M, N and K must be positive; K must be a multiple of 256");
    }
    if (warmup < 0 || iterations <= 0) {
      throw std::invalid_argument("iterations must be positive and warmup non-negative");
    }
  }
};

struct Tensors {
  using DataA = ElementA::DataType;
  using DataB = ElementB::DataType;
  using ScaleA = ElementA::ScaleFactorType;
  using ScaleB = ElementB::ScaleFactorType;
  using VectorLayout = cutlass::layout::PackedVectorLayout;
  using StrideA = Gemm::GemmKernel::StrideA;
  using StrideB = Gemm::GemmKernel::StrideB;
  using StrideC = Gemm::GemmKernel::StrideC;
  using StrideD = Gemm::GemmKernel::StrideD;
  using LayoutSFA = Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

  cutlass::HostTensor<DataA, VectorLayout> a;
  cutlass::HostTensor<DataB, VectorLayout> b;
  cutlass::HostTensor<ScaleA, VectorLayout> sfa;
  cutlass::HostTensor<ScaleB, VectorLayout> sfb;
  cutlass::HostTensor<ElementC, VectorLayout> c;
  cutlass::HostTensor<ElementD, VectorLayout> d;
  cutlass::HostTensor<ElementD, VectorLayout> reference;
  StrideA stride_a;
  StrideB stride_b;
  StrideC stride_c;
  StrideD stride_d;
  LayoutSFA layout_sfa;
  LayoutSFB layout_sfb;
  Gemm::Arguments arguments;

  explicit Tensors(Options const& options) {
    using Config = Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    stride_a = cutlass::make_cute_packed_stride<StrideA>({options.m, options.k, 1});
    stride_b = cutlass::make_cute_packed_stride<StrideB>({options.n, options.k, 1});
    stride_c = cutlass::make_cute_packed_stride<StrideC>({options.m, options.n, 1});
    stride_d = cutlass::make_cute_packed_stride<StrideD>({options.m, options.n, 1});
    auto layout_a = make_layout(make_shape(options.m, options.k, 1), stride_a);
    auto layout_b = make_layout(make_shape(options.n, options.k, 1), stride_b);
    auto layout_c = make_layout(make_shape(options.m, options.n, 1), stride_c);
    auto layout_d = make_layout(make_shape(options.m, options.n, 1), stride_d);
    layout_sfa = Config::tile_atom_to_shape_SFA({options.m, options.n, options.k, 1});
    layout_sfb = Config::tile_atom_to_shape_SFB({options.m, options.n, options.k, 1});
    a.reset(cutlass::make_Coord(size(layout_a)));
    b.reset(cutlass::make_Coord(size(layout_b)));
    c.reset(cutlass::make_Coord(size(layout_c)));
    d.reset(cutlass::make_Coord(size(layout_d)));
    reference.reset(cutlass::make_Coord(size(layout_d)));
    sfa.reset(cutlass::make_Coord(size(filter_zeros(layout_sfa))));
    sfb.reset(cutlass::make_Coord(size(filter_zeros(layout_sfb))));
    cutlass::reference::host::TensorFillRandomUniform(a.host_view(), 17, 2.0, -2.0, 4);
    cutlass::reference::host::TensorFillRandomUniform(b.host_view(), 23, 2.0, -2.0, 4);
    cutlass::reference::host::TensorFillRandomUniform(sfa.host_view(), 29, 1.0, 0.25, 0);
    cutlass::reference::host::TensorFillRandomUniform(sfb.host_view(), 31, 1.0, 0.25, 0);
    cutlass::reference::host::TensorFill(c.host_view(), ElementC(0.0f));
    a.sync_device();
    b.sync_device();
    sfa.sync_device();
    sfb.sync_device();
    c.sync_device();
    d.sync_device();
    arguments = Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {options.m, options.n, options.k, 1},
        {a.device_data(), stride_a, b.device_data(), stride_b,
         sfa.device_data(), layout_sfa, sfb.device_data(), layout_sfb},
        {{1.0f, 0.0f}, c.device_data(), stride_c, d.device_data(), stride_d}};
    saved_layout_a = layout_a;
    saved_layout_b = layout_b;
    saved_layout_c = layout_c;
    saved_layout_d = layout_d;
  }

  bool verify() {
    auto tensor_a = make_tensor(cute::recast_ptr<DataA>(a.host_data()), saved_layout_a);
    auto tensor_b = make_tensor(cute::recast_ptr<DataB>(b.host_data()), saved_layout_b);
    auto tensor_sfa = make_tensor(sfa.host_data(), layout_sfa);
    auto tensor_sfb = make_tensor(sfb.host_data(), layout_sfb);
    auto tensor_c = make_tensor(cute::recast_ptr<ElementC>(c.host_data()), saved_layout_c);
    auto tensor_d = make_tensor(cute::recast_ptr<ElementD>(reference.host_data()), saved_layout_d);
    cutlass::reference::host::GettBlockScalingMainloopParams<
        float, decltype(tensor_a), decltype(tensor_sfa), decltype(tensor_b), decltype(tensor_sfb)>
        mainloop{tensor_a, tensor_sfa, tensor_b, tensor_sfb};
    cutlass::reference::host::GettEpilogueParams<
        float, float, float, float, decltype(tensor_c), decltype(tensor_d)>
        epilogue{1.0f, 0.0f, tensor_c, tensor_d};
    cutlass::reference::host::Gemm3x(mainloop, epilogue);
    d.sync_host();
    return cutlass::reference::host::TensorEquals(reference.host_view(), d.host_view());
  }

 private:
  decltype(make_layout(make_shape(0, 0, 0), StrideA{})) saved_layout_a;
  decltype(make_layout(make_shape(0, 0, 0), StrideB{})) saved_layout_b;
  decltype(make_layout(make_shape(0, 0, 0), StrideC{})) saved_layout_c;
  decltype(make_layout(make_shape(0, 0, 0), StrideD{})) saved_layout_d;
};

int run(Options const& options) {
  Tensors tensors(options);
  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(tensors.arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  CUTLASS_CHECK(gemm.can_implement(tensors.arguments));
  CUTLASS_CHECK(gemm.initialize(tensors.arguments, workspace.get()));
  CUTLASS_CHECK(gemm.run());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (options.verify && !tensors.verify()) {
    std::cerr << "CUTLASS output differs from the NVFP4 host reference\n";
    return 3;
  }
  for (int i = 0; i < options.warmup; ++i) CUTLASS_CHECK(gemm.run());
  GpuTimer timer;
  timer.start();
  for (int i = 0; i < options.iterations; ++i) CUTLASS_CHECK(gemm.run());
  timer.stop();
  double elapsed_ms = timer.elapsed_millis() / options.iterations;
  double flops = 2.0 * options.m * options.n * options.k;
  std::cout << "verified=" << (options.verify ? "true" : "false") << "\n"
            << "m=" << options.m << " n=" << options.n << " k=" << options.k << "\n"
            << "nvfp4_ms=" << elapsed_ms << "\n"
            << "nvfp4_tflops=" << flops / elapsed_ms * 1e-9 << "\n"
            << "gemm_tokens_per_second=" << options.m / elapsed_ms * 1000.0 << "\n";
  return 0;
}
#endif

int main(int argc, char const** argv) {
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "CUDA 12.8 or newer is required\n";
    return 2;
  }
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
  if (properties.major != 10 || (properties.minor != 0 && properties.minor != 1 && properties.minor != 3)) {
    std::cerr << "This executable requires datacenter Blackwell SM100/SM103\n";
    return 2;
  }
  try {
    Options options;
    options.parse(argc, argv);
    return run(options);
  } catch (std::exception const& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
#else
  std::cerr << "Rebuild with SM100 block-scaled MMA support enabled\n";
  return 2;
#endif
}
