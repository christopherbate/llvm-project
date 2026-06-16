//===------------------ TestVulkanRunnerPipeline.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements a pipeline for use by Vulkan runner tests.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

// Defined in the test directory, no public header.
namespace mlir::test {
std::unique_ptr<Pass> createTestConvertToSPIRVPass(bool convertGPUModules,
                                                   bool nestInGPUModule);
} // namespace mlir::test

namespace {

// Sentinel the Vulkan runtime wrapper uses to select the buffer-device-address
// launch path. Must match kVulkanBufferDeviceAddressMagic in
// VulkanRuntimeWrappers.cpp.
static constexpr uint64_t kVulkanBufferDeviceAddressMagic =
    0x4244414d4c524956ULL;

// Sentinel selecting the raw-argument (CUDA-style) launch path. Must match
// kVulkanRawArgsMagic in VulkanRuntimeWrappers.cpp.
static constexpr uint64_t kVulkanRawArgsMagic = 0x5752414d4c524956ULL;

// Rewrites a `gpu.launch_func` with memref kernel operands into the
// buffer-device-address calling convention understood by the Vulkan runtime.
//
// Each memref operand is passed using MLIR's standard *unranked* memref
// descriptor ({ i64 rank, ptr to ranked descriptor }), alongside its data size
// in bytes. The runtime reads the ranked descriptor directly and builds the
// device-resident descriptor itself. The rewritten launch operand list is:
//
//   [ marker, (unrankedDescriptor, byteSize) x numBuffers ]
struct ConvertLaunchFuncToBufferDeviceAddress
    : public ConvertOpToLLVMPattern<gpu::LaunchFuncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    Type i64Ty = rewriter.getI64Type();
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto unrankedTy = LLVM::LLVMStructType::getLiteral(ctx, {i64Ty, ptrTy});

    SmallVector<MemRefType> memrefTypes;
    for (Value operand : op.getKernelOperands()) {
      auto memrefTy = dyn_cast<MemRefType>(operand.getType());
      if (!memrefTy)
        return rewriter.notifyMatchFailure(op, "non-memref kernel operand");
      memrefTypes.push_back(memrefTy);
    }

    auto constI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                      rewriter.getI64IntegerAttr(v));
    };

    SmallVector<Value> kernelOperands;
    kernelOperands.push_back(
        constI64(static_cast<int64_t>(kVulkanBufferDeviceAddressMagic)));

    for (auto [k, memrefTy] : llvm::enumerate(memrefTypes)) {
      Value rankedDesc = adaptor.getKernelOperands()[k];
      MemRefDescriptor desc(rankedDesc);
      int64_t rank = memrefTy.getRank();

      // Store the ranked descriptor to the stack and build the standard
      // unranked descriptor { rank, ptr } referencing it.
      Value rankedDescPtr = LLVM::AllocaOp::create(
          rewriter, loc, ptrTy, rankedDesc.getType(), constI64(1));
      LLVM::StoreOp::create(rewriter, loc, rankedDesc, rankedDescPtr);
      Value unranked = LLVM::PoisonOp::create(rewriter, loc, unrankedTy);
      unranked = LLVM::InsertValueOp::create(rewriter, loc, unranked,
                                             constI64(rank), ArrayRef<int64_t>{0});
      unranked = LLVM::InsertValueOp::create(rewriter, loc, unranked,
                                             rankedDescPtr, ArrayRef<int64_t>{1});

      // Data size in bytes = product(sizes) * sizeof(element). getNumElements
      // folds static dims to constants and only reads the dynamic sizes from
      // the descriptor; getSizeInBytes uses the GEP-on-null trick so the
      // element size follows the data layout.
      SmallVector<Value> dynamicSizes;
      for (int64_t d = 0; d < rank; ++d)
        if (memrefTy.isDynamicDim(d))
          dynamicSizes.push_back(desc.size(rewriter, loc, d));
      Value numElements =
          getNumElements(loc, memrefTy, dynamicSizes, rewriter);
      Value elementBytes =
          getSizeInBytes(loc, memrefTy.getElementType(), rewriter);
      Value byteSize =
          LLVM::MulOp::create(rewriter, loc, numElements, elementBytes);

      kernelOperands.push_back(unranked);
      kernelOperands.push_back(byteSize);
    }

    gpu::LaunchFuncOp::create(
        rewriter, loc, op.getKernelAttr(),
        gpu::KernelDim3{adaptor.getGridSizeX(), adaptor.getGridSizeY(),
                        adaptor.getGridSizeZ()},
        gpu::KernelDim3{adaptor.getBlockSizeX(), adaptor.getBlockSizeY(),
                        adaptor.getBlockSizeZ()},
        adaptor.getDynamicSharedMemorySize(), kernelOperands);
    rewriter.eraseOp(op);
    return success();
  }
};

// Pass driving the buffer-device-address launch conversion.
struct ConvertVulkanLaunchToBufferDeviceAddressPass
    : public PassWrapper<ConvertVulkanLaunchToBufferDeviceAddressPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConvertVulkanLaunchToBufferDeviceAddressPass)

  void runOnOperation() override {
    LLVMTypeConverter converter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertLaunchFuncToBufferDeviceAddress>(converter);

    ConversionTarget target(getContext());
    // A launch is illegal until its memref kernel operands have been replaced
    // with the buffer-device-address encoding (all non-memref operands).
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>([](gpu::LaunchFuncOp op) {
      return llvm::none_of(op.getKernelOperands(), [](Value v) {
        return isa<MemRefType>(v.getType());
      });
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

// Rewrites every `gpu.launch_func` whose operands are by-value scalars and
// pointers (the CUDA-style raw-argument convention) into the raw-args calling
// convention understood by the Vulkan runtime: the operands are packed, in
// order, into a host param blob (one 8-byte slot each) that the runtime pushes
// as the kernel's push constants. Device addresses (e.g. from mgpuMemAlloc) are
// just integer operands here; the kernel dereferences them via
// PhysicalStorageBuffer. The rewritten operand list is
// [ marker, blobPtr, blobByteSize ].
//
// This runs just before the generic gpu-to-llvm lowering, when the launch
// operands are already LLVM-storable scalar/pointer values, so it uses a direct
// rewrite rather than a type conversion.
static LogicalResult rewriteLaunchToRawArgs(gpu::LaunchFuncOp op) {
  for (Value operand : op.getKernelOperands())
    if (isa<MemRefType>(operand.getType()))
      return op.emitError("raw-args launch does not support memref operands");

  IRRewriter rewriter(op);
  rewriter.setInsertionPoint(op);
  Location loc = op.getLoc();
  Type i64Ty = rewriter.getI64Type();
  auto ptrTy = LLVM::LLVMPointerType::get(op.getContext());

  auto constI64 = [&](int64_t v) -> Value {
    return LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                    rewriter.getI64IntegerAttr(v));
  };

  unsigned numArgs = op.getKernelOperands().size();
  auto arrayTy = LLVM::LLVMArrayType::get(i64Ty, numArgs ? numArgs : 1);
  Value blob =
      LLVM::AllocaOp::create(rewriter, loc, ptrTy, arrayTy, constI64(1));
  for (auto [k, operand] : llvm::enumerate(op.getKernelOperands())) {
    Value value = operand;
    if (isa<LLVM::LLVMPointerType>(value.getType()))
      value = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, value);
    SmallVector<LLVM::GEPArg> indices{0, static_cast<int32_t>(k)};
    Value slot =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, arrayTy, blob, indices);
    LLVM::StoreOp::create(rewriter, loc, value, slot);
  }

  SmallVector<Value> kernelOperands{
      constI64(static_cast<int64_t>(kVulkanRawArgsMagic)), blob,
      constI64(numArgs * static_cast<int64_t>(sizeof(uint64_t)))};

  gpu::LaunchFuncOp::create(
      rewriter, loc, op.getKernelAttr(),
      gpu::KernelDim3{op.getGridSizeX(), op.getGridSizeY(), op.getGridSizeZ()},
      gpu::KernelDim3{op.getBlockSizeX(), op.getBlockSizeY(),
                      op.getBlockSizeZ()},
      op.getDynamicSharedMemorySize(), kernelOperands);
  rewriter.eraseOp(op);
  return success();
}

// Pass driving the raw-argument launch conversion.
struct ConvertVulkanLaunchToRawArgsPass
    : public PassWrapper<ConvertVulkanLaunchToRawArgsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertVulkanLaunchToRawArgsPass)

  void runOnOperation() override {
    SmallVector<gpu::LaunchFuncOp> launches;
    getOperation().walk(
        [&](gpu::LaunchFuncOp op) { launches.push_back(op); });
    for (gpu::LaunchFuncOp op : launches)
      if (failed(rewriteLaunchToRawArgs(op)))
        return signalPassFailure();
  }
};

struct VulkanRunnerPipelineOptions
    : PassPipelineOptions<VulkanRunnerPipelineOptions> {
  Option<bool> spirvWebGPUPrepare{
      *this, "spirv-webgpu-prepare",
      llvm::cl::desc("Run MLIR transforms used when targetting WebGPU")};
  Option<bool> kernelIsSPIRV{
      *this, "kernel-is-spirv",
      llvm::cl::desc("Assume the kernel is already written in the SPIR-V "
                     "dialect (skip the GPU-to-SPIR-V conversion). The "
                     "gpu.module must carry its own #spirv.target_env target.")};
  Option<bool> useBufferDeviceAddress{
      *this, "use-buffer-device-address",
      llvm::cl::desc("Lower kernel launches to the buffer-device-address "
                     "(PhysicalStorageBuffer) calling convention: pass memref "
                     "buffers to the kernel through device-resident descriptors "
                     "in an argument buffer instead of descriptor bindings.")};
  Option<bool> useRawArgs{
      *this, "use-raw-args",
      llvm::cl::desc("Lower kernel launches to the raw-argument (CUDA-style) "
                     "calling convention: pass by-value scalars and device "
                     "pointers (e.g. from mgpuMemAlloc) directly as push "
                     "constants, with no runtime-managed buffers.")};
};

void buildTestVulkanRunnerPipeline(OpPassManager &passManager,
                                   const VulkanRunnerPipelineOptions &options) {
  passManager.addPass(createGpuKernelOutliningPass());
  passManager.addPass(memref::createFoldMemRefAliasOpsPass());

  // When the kernel is hand-written directly in the SPIR-V dialect (e.g. to
  // exercise features the GPU-to-SPIR-V conversion does not produce, such as
  // the PhysicalStorageBuffer addressing model), skip the conversion passes and
  // serialize the existing spirv.module as-is.
  if (!options.kernelIsSPIRV) {
    GpuSPIRVAttachTargetOptions attachTargetOptions{};
    attachTargetOptions.spirvVersion = "v1.0";
    attachTargetOptions.spirvCapabilities.push_back("Shader");
    attachTargetOptions.spirvExtensions.push_back(
        "SPV_KHR_storage_buffer_storage_class");
    passManager.addPass(createGpuSPIRVAttachTarget(attachTargetOptions));

    passManager.addPass(test::createTestConvertToSPIRVPass(
        /*convertGPUModules=*/true, /*nestInGPUModule=*/true));

    OpPassManager &spirvModulePM =
        passManager.nest<gpu::GPUModuleOp>().nest<spirv::ModuleOp>();
    spirvModulePM.addPass(spirv::createSPIRVLowerABIAttributesPass());
    spirvModulePM.addPass(spirv::createSPIRVUpdateVCEPass());
    if (options.spirvWebGPUPrepare)
      spirvModulePM.addPass(spirv::createSPIRVWebGPUPreparePass());
  }

  passManager.addPass(createGpuModuleToBinaryPass());

  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  passManager.nest<func::FuncOp>().addPass(
      LLVM::createLLVMRequestCWrappersPass());

  if (options.useRawArgs) {
    // Encode launches into the raw-argument calling convention before the
    // generic gpu-to-llvm lowering turns them into runtime calls.
    passManager.addPass(std::make_unique<ConvertVulkanLaunchToRawArgsPass>());
    passManager.addPass(createGpuToLLVMConversionPass());
  } else if (options.useBufferDeviceAddress) {
    // Encode launches into the buffer-device-address calling convention before
    // the generic gpu-to-llvm lowering turns them into runtime calls. The
    // generic lowering then runs with default options (no bare-pointer / size
    // interspersing), passing the already-encoded operands through unchanged.
    passManager.addPass(
        std::make_unique<ConvertVulkanLaunchToBufferDeviceAddressPass>());
    passManager.addPass(createGpuToLLVMConversionPass());
  } else {
    // VulkanRuntimeWrappers.cpp requires these calling convention options.
    GpuToLLVMConversionPassOptions opt;
    opt.hostBarePtrCallConv = false;
    opt.kernelBarePtrCallConv = true;
    opt.kernelIntersperseSizeCallConv = true;
    passManager.addPass(createGpuToLLVMConversionPass(opt));
  }
  passManager.addPass(createReconcileUnrealizedCastsPass());
}

} // namespace

namespace mlir::test {
void registerTestVulkanRunnerPipeline() {
  PassPipelineRegistration<VulkanRunnerPipelineOptions>(
      "test-vulkan-runner-pipeline",
      "Runs a series of passes intended for Vulkan runner tests. Lowers GPU "
      "dialect to LLVM dialect for the host and to serialized Vulkan SPIR-V "
      "for the device.",
      buildTestVulkanRunnerPipeline);
}
} // namespace mlir::test
