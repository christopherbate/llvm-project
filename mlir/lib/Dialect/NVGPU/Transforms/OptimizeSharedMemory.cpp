//===- OptimizeSharedMemory.cpp - MLIR operations for math implementation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements transforms to optimize accesses to shared memory.
//
//===----------------------------------------------------------------------===//
#include "PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Passes.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::nvgpu;

/// The size of a shared memory line according to NV documentation.
// TODO: insert Nvidia docs link.
constexpr int64_t kSharedMemoryLineSizeBytes = 128;
constexpr int64_t kDefaultVectorSizeBits = 128;

/// Uses `srcIndexValue` to permute `tgtIndexValue` via
/// `result = xor(floordiv(srcIdxVal,permuteEveryN),
///               floordiv(tgtIdxVal,vectorSize)))
///            + tgtIdxVal % vectorSize`
/// This is done using an optimized sequence of `arith` operations.
static Value permuteVectorOffset(OpBuilder &b, Location loc,
                                 ArrayRef<Value> indices, MemRefType memrefTy,
                                 int64_t srcDim, int64_t tgtDim) {
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);

  // Adjust the src index to change how often the permutation changes
  // if necessary.
  Value src = indices[srcDim];

  // We only want to permute every N iterations of the target dim where N is
  // ceil(sharedMemoryLineSizeBytes / dimSizeBytes(tgtDim)).
  const int64_t permuteEveryN = std::max<int64_t>(
      1, kSharedMemoryLineSizeBytes / ((memrefTy.getDimSize(tgtDim) *
                                        memrefTy.getElementTypeBitWidth()) /
                                       8));

  // clang-format off
  // Index bit representation (b0 = least significant bit) for dim(1)
  // of a `memref<?x?xDT>` is as follows:
  // N := log2(128/elementSizeBits)
  // M := log2(dimSize(1))
  // then
  // bits[0:N] = sub-vector element offset
  // bits[N:M] = vector index
  // clang-format on
  int64_t N =
      llvm::Log2_64(kDefaultVectorSizeBits / memrefTy.getElementTypeBitWidth());
  int64_t M = llvm::Log2_64(memrefTy.getDimSize(tgtDim));

  // Capture bits[0:(M-N)] of src by first creating a (M-N) mask.
  int64_t mask = (1 << (M - N)) - 1;
  if (permuteEveryN > 1)
    mask = mask << llvm::Log2_64(permuteEveryN);
  Value srcBits = b.create<arith::ConstantIndexOp>(loc, mask);
  srcBits = b.create<arith::AndIOp>(loc, src, srcBits);

  // Use the src bits to permute the target bits b[N:M].
  auto nval = b.create<arith::ConstantIndexOp>(
      loc, permuteEveryN > 1 ? N - llvm::Log2_64(permuteEveryN) : N);
  srcBits = b.createOrFold<arith::ShLIOp>(loc, srcBits, nval);
  Value permutedVectorIdx =
      b.create<arith::XOrIOp>(loc, indices[tgtDim], srcBits);
  return permutedVectorIdx;
}

static void transformIndices(OpBuilder &builder, Location loc,
                             SmallVector<Value, 4> &indices,
                             MemRefType memrefTy, int64_t srcDim,
                             int64_t tgtDim) {
  indices[tgtDim] =
      permuteVectorOffset(builder, loc, indices, memrefTy, srcDim, tgtDim);
}

Operation::operand_range getIndices(Operation *op) {
  if (auto ldmatrixOp = dyn_cast<LdMatrixOp>(op))
    return ldmatrixOp.indices();
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.indices();
  if (auto vectorReadOp = dyn_cast<vector::TransferReadOp>(op))
    return vectorReadOp.getIndices();
  if (auto copyOp = dyn_cast<DeviceAsyncCopyOp>(op))
    return copyOp.dstIndices();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.indices();
  llvm_unreachable("unsupported op type");
}

/// Return all operations within `parentOp` that read from or write to
/// `shmMemRef`.
static LogicalResult
getShmReadAndWriteOps(Operation *parentOp, Value shmMemRef,
                      SmallVector<Operation *, 16> &readOps,
                      SmallVector<Operation *, 16> &writeOps) {
  parentOp->walk([&](Operation *op) {
    MemoryEffectOpInterface iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface)
      return;
    Optional<MemoryEffects::EffectInstance> effect =
        iface.getEffectOnValue<MemoryEffects::Read>(shmMemRef);
    if (effect) {
      readOps.push_back(op);
      return;
    }
    effect = iface.getEffectOnValue<MemoryEffects::Write>(shmMemRef);
    if (effect)
      writeOps.push_back(op);
  });

  // Restrict to a supported set of ops. We also require at least 2D access,
  // although this could be relaxed.
  if (llvm::any_of(readOps, [](Operation *op) {
        return !isa<memref::LoadOp, vector::LoadOp, nvgpu::LdMatrixOp>(op) ||
               getIndices(op).size() < 2;
      }))
    return failure();
  if (llvm::any_of(writeOps, [](Operation *op) {
        return !isa<vector::StoreOp, nvgpu::DeviceAsyncCopyOp>(op) ||
               getIndices(op).size() < 2;
      }))
    return failure();

  return success();
}

mlir::LogicalResult
mlir::nvgpu::optimizeSharedMemoryReadsAndWrites(Operation *parentOp,
                                                Value memrefValue) {
  auto memRefType = memrefValue.getType().dyn_cast<MemRefType>();
  if (!memRefType ||
      memRefType.getMemorySpaceAsInt() != NVVM::kSharedMemorySpace)
    return failure();

  // Get sets of operations within the function that read/write to shared
  // memory.
  SmallVector<Operation *, 16> shmReadOps;
  SmallVector<Operation *, 16> shmWriteOps;
  if (failed(getShmReadAndWriteOps(parentOp, memrefValue, shmReadOps,
                                   shmWriteOps)))
    return failure();

  if (shmReadOps.empty() || shmWriteOps.empty())
    return failure();

  OpBuilder builder(parentOp->getContext());

  int64_t tgtDim = memRefType.getRank() - 1;
  int64_t srcDim = memRefType.getRank() - 2;

  // Transform indices for the ops writing to shared memory.
  while (!shmWriteOps.empty()) {
    Operation *shmWriteOp = shmWriteOps.back();
    shmWriteOps.pop_back();
    builder.setInsertionPoint(shmWriteOp);

    auto indices = getIndices(shmWriteOp);
    SmallVector<Value, 4> transformedIndices(indices.begin(), indices.end());
    transformIndices(builder, shmWriteOp->getLoc(), transformedIndices,
                     memRefType, srcDim, tgtDim);
    shmWriteOp->setOperands(1, transformedIndices.size(), transformedIndices);
    shmWriteOp->setAttr("optimized_store", builder.getUnitAttr());
  }

  // Transform indices for the ops reading from shared memory.
  while (!shmReadOps.empty()) {
    Operation *shmReadOp = shmReadOps.back();
    shmReadOps.pop_back();
    builder.setInsertionPoint(shmReadOp);

    auto indices = getIndices(shmReadOp);
    SmallVector<Value, 4> transformedIndices(indices.begin(), indices.end());
    transformIndices(builder, shmReadOp->getLoc(), transformedIndices,
                     memRefType, srcDim, tgtDim);
    shmReadOp->setOperands(1, indices.size(), transformedIndices);
    shmReadOp->setAttr("optimized_load", builder.getUnitAttr());
  }

  return success();
}

namespace {
class OptimizeSharedMemoryPass
    : public OptimizeSharedMemoryBase<OptimizeSharedMemoryPass> {
public:
  OptimizeSharedMemoryPass() = default;

  void runOnOperation() override {
    Operation *op = getOperation();
    SmallVector<memref::AllocOp> shmAllocOps;
    op->walk([&](memref::AllocOp allocOp) {
      if (allocOp.memref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
          NVVM::kSharedMemorySpace)
        return;
      shmAllocOps.push_back(allocOp);
    });
    for (auto allocOp : shmAllocOps) {
      if (failed(optimizeSharedMemoryReadsAndWrites(getOperation(),
                                                    allocOp.memref())))
        return;
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::nvgpu::createOptimizeSharedMemoryPass() {
  return std::make_unique<OptimizeSharedMemoryPass>();
}