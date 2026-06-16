// Validates the host-side device memory API (mgpuMemAlloc / mgpuMemcpy /
// mgpuMemFree) in isolation, with no kernel launch: host data is copied to a
// device allocation and back, and must round-trip unchanged. This is the
// host-owned device memory that the buffer-device-address / raw-pointer kernel
// launch path relies on.

// RUN: mlir-opt %s -test-vulkan-runner-pipeline \
// RUN:   | mlir-runner - --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils --entry-point-result=void | FileCheck %s

// CHECK: [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
module {
  func.func @main() {
    %c32 = arith.constant 32 : i64
    %fill = arith.constant 1.5 : f32
    %zero = arith.constant 0.0 : f32

    // Host source buffer, filled with 1.5.
    %src = memref.alloc() : memref<8xf32>
    %src_dyn = memref.cast %src : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%src_dyn, %fill) : (memref<?xf32>, f32) -> ()

    // Host destination buffer, zeroed.
    %dst = memref.alloc() : memref<8xf32>
    %dst_dyn = memref.cast %dst : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%dst_dyn, %zero) : (memref<?xf32>, f32) -> ()

    // Raw host pointers as integers.
    %src_idx = memref.extract_aligned_pointer_as_index %src : memref<8xf32> -> index
    %src_ptr = arith.index_cast %src_idx : index to i64
    %dst_idx = memref.extract_aligned_pointer_as_index %dst : memref<8xf32> -> index
    %dst_ptr = arith.index_cast %dst_idx : index to i64

    // Allocate device memory, copy host->device->host through it.
    %dev = call @mgpuMemAlloc(%c32) : (i64) -> i64
    call @mgpuMemcpyHostToDevice(%dev, %src_ptr, %c32) : (i64, i64, i64) -> ()
    call @mgpuMemcpyDeviceToHost(%dst_ptr, %dev, %c32) : (i64, i64, i64) -> ()
    call @mgpuMemFree(%dev) : (i64) -> ()

    %dst_unranked = memref.cast %dst : memref<8xf32> to memref<*xf32>
    call @printMemrefF32(%dst_unranked) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
  func.func private @mgpuMemAlloc(%size : i64) -> i64
  func.func private @mgpuMemcpyHostToDevice(%dst : i64, %src : i64, %size : i64)
  func.func private @mgpuMemcpyDeviceToHost(%dst : i64, %src : i64, %size : i64)
  func.func private @mgpuMemFree(%addr : i64)
}
