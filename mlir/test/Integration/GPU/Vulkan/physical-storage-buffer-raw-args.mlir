// Demonstrates the CUDA-style raw-argument launch path: the kernel takes a raw
// device pointer (an i64 buffer device address) and a by-value scalar, instead
// of bound memref descriptors. The host allocates device memory with the memory
// API (mgpuMemAlloc / mgpuMemcpy), passes the device address + scalar directly
// to the kernel, and reads the result back. The launch pass packs the by-value
// operands into push constants; the kernel dereferences the address via
// PhysicalStorageBuffer and adds the scalar (computing out[i] = in[i] + n).

// RUN: mlir-opt %s -test-vulkan-runner-pipeline='kernel-is-spirv use-raw-args' \
// RUN:   | mlir-runner - --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils --entry-point-result=void | FileCheck %s

// CHECK: [11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5]
module attributes {gpu.container_module} {
  gpu.module @kernels [#spirv.target_env<
      #spirv.vce<v1.5, [Shader, Int64, PhysicalStorageBufferAddresses],
                 [SPV_KHR_physical_storage_buffer]>,
      #spirv.resource_limits<>>] {
    // Placeholder kernel matching the launch signature (raw device address +
    // scalar). The serialized SPIR-V below is what actually runs.
    gpu.func @add_scalar(%arg0 : i64, %arg1 : i32) kernel {
      gpu.return
    }

    spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.5,
        [Shader, Int64, PhysicalStorageBufferAddresses],
        [SPV_KHR_physical_storage_buffer]> {
      // Push constant block holding the kernel's by-value arguments: the buffer
      // device address (i64 at offset 0) and the scalar (i32 at offset 8).
      spirv.GlobalVariable @pc : !spirv.ptr<!spirv.struct<(i64 [0], i32 [8])>, PushConstant>
      spirv.GlobalVariable @gid built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
      spirv.func @add_scalar() "None" {
        %c0 = spirv.Constant 0 : i32
        %c1 = spirv.Constant 1 : i32
        %pc = spirv.mlir.addressof @pc : !spirv.ptr<!spirv.struct<(i64 [0], i32 [8])>, PushConstant>
        %addr_p = spirv.AccessChain %pc[%c0] : !spirv.ptr<!spirv.struct<(i64 [0], i32 [8])>, PushConstant>, i32 -> !spirv.ptr<i64, PushConstant>
        %addr = spirv.Load "PushConstant" %addr_p : i64
        %n_p = spirv.AccessChain %pc[%c1] : !spirv.ptr<!spirv.struct<(i64 [0], i32 [8])>, PushConstant>, i32 -> !spirv.ptr<i32, PushConstant>
        %n = spirv.Load "PushConstant" %n_p : i32
        %nf = spirv.ConvertSToF %n : i32 to f32
        // Reinterpret the raw address as a PhysicalStorageBuffer pointer.
        %data = spirv.ConvertUToPtr %addr : i64 to !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>
        %gid_a = spirv.mlir.addressof @gid : !spirv.ptr<vector<3xi32>, Input>
        %gid_v = spirv.Load "Input" %gid_a : vector<3xi32>
        %tid = spirv.CompositeExtract %gid_v[0 : i32] : vector<3xi32>
        %elem = spirv.AccessChain %data[%c0, %tid] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>, i32, i32 -> !spirv.ptr<f32, PhysicalStorageBuffer>
        %v = spirv.Load "PhysicalStorageBuffer" %elem ["Aligned", 4] : f32
        %r = spirv.FAdd %v, %nf : f32
        spirv.Store "PhysicalStorageBuffer" %elem, %r ["Aligned", 4] : f32
        spirv.Return
      }
      spirv.EntryPoint "GLCompute" @add_scalar, @gid, @pc
      spirv.ExecutionMode @add_scalar "LocalSize", 1, 1, 1
    }
  }

  func.func @main() {
    %bytes = arith.constant 32 : i64
    %n = arith.constant 10 : i32
    %fill = arith.constant 1.5 : f32

    // Host buffer filled with 1.5.
    %host = memref.alloc() : memref<8xf32>
    %host_dyn = memref.cast %host : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%host_dyn, %fill) : (memref<?xf32>, f32) -> ()
    %host_idx = memref.extract_aligned_pointer_as_index %host : memref<8xf32> -> index
    %host_ptr = arith.index_cast %host_idx : index to i64

    // Allocate device memory and upload.
    %dev = call @mgpuMemAlloc(%bytes) : (i64) -> i64
    call @mgpuMemcpyHostToDevice(%dev, %host_ptr, %bytes) : (i64, i64, i64) -> ()

    // Launch with the raw device address + scalar as by-value arguments.
    %cst1 = arith.constant 1 : index
    %cst8 = arith.constant 8 : index
    gpu.launch_func @kernels::@add_scalar
        blocks in (%cst8, %cst1, %cst1) threads in (%cst1, %cst1, %cst1)
        args(%dev : i64, %n : i32)

    // Read back and free.
    call @mgpuMemcpyDeviceToHost(%host_ptr, %dev, %bytes) : (i64, i64, i64) -> ()
    call @mgpuMemFree(%dev) : (i64) -> ()

    %host_unranked = memref.cast %host : memref<8xf32> to memref<*xf32>
    call @printMemrefF32(%host_unranked) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
  func.func private @mgpuMemAlloc(%size : i64) -> i64
  func.func private @mgpuMemcpyHostToDevice(%dst : i64, %src : i64, %size : i64)
  func.func private @mgpuMemcpyDeviceToHost(%dst : i64, %src : i64, %size : i64)
  func.func private @mgpuMemFree(%addr : i64)
}
