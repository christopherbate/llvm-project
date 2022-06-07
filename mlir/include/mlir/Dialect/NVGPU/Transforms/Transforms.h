//===- Transforms.h - NVGPU Dialect transformations --------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions that assist transformations for the nvgpu
// dialect.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_NVGPU_TRANSFORMS_TRANSFORMS_H_
#define MLIR_DIALECT_NVGPU_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace nvgpu {

/// Optimizes vectorized accesses to a shared memory buffer specified by
/// memrefValue. This transformation assumes the following:
/// 1) All relevant accesses to `memrefValue` are contained with `parentOp`.
/// 2) The function will fail precondition checks if any subviews are
/// taken of `memrefValue`. All reads/writes to `memrefValue` should occur
/// through `memrefValue` directly.
mlir::LogicalResult optimizeSharedMemoryReadsAndWrites(Operation *parentOp,
                                                       Value memrefValue);

} // namespace nvgpu
} // namespace mlir

#endif // MLIR_DIALECT_NVGPU_TRANSFORMS_TRANSFORMS_H_
