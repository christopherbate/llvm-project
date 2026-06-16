// Demonstrates the Vulkan runtime's support for the PhysicalStorageBuffer
// addressing model (a.k.a. buffer device address). The compute kernel is
// written directly in the SPIR-V dialect. Instead of receiving its operand as a
// bound StorageBuffer descriptor, the kernel receives the device address of an
// *argument buffer* (as the sole push constant) which holds a device-resident
// MLIR ranked memref descriptor ({allocatedPtr, alignedPtr, offset, sizes,
// strides}, all i64, with the pointer fields holding the buffer's device
// address). The runtime stays type-agnostic: the compiler hands it memrefs via
// the standard unranked descriptor convention, and the launch selects the
// buffer-device-address path via a marker (the runtime never inspects the
// SPIR-V).

// RUN: mlir-opt %s -test-vulkan-runner-pipeline='kernel-is-spirv use-buffer-device-address' \
// RUN:   | mlir-runner - --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils --entry-point-result=void | FileCheck %s

// CHECK: [3, 3, 3, 3, 3, 3, 3, 3]
module attributes {gpu.container_module} {
  gpu.module @kernels [#spirv.target_env<
      #spirv.vce<v1.5, [Shader, Int64, PhysicalStorageBufferAddresses],
                 [SPV_KHR_physical_storage_buffer]>,
      #spirv.resource_limits<>>] {
    gpu.func @double_buffer(%arg0 : memref<8xf32>) kernel {
      gpu.return
    }

    spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.5,
        [Shader, Int64, PhysicalStorageBufferAddresses],
        [SPV_KHR_physical_storage_buffer]> {
      spirv.GlobalVariable @pc : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>
      spirv.GlobalVariable @gid built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
      spirv.func @double_buffer() "None" {
        %c0 = spirv.Constant 0 : i32
        %c1 = spirv.Constant 1 : i32
        // Root pointer (device address of the argument buffer) from the push constant.
        %pc = spirv.mlir.addressof @pc : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>
        %root_ptr = spirv.AccessChain %pc[%c0] : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>, i32 -> !spirv.ptr<i64, PushConstant>
        %root = spirv.Load "PushConstant" %root_ptr : i64
        // Ranked descriptor { allocatedPtr, alignedPtr, offset, size, stride };
        // read the alignedPtr (field 1) as the data device address.
        %desc = spirv.ConvertUToPtr %root : i64 to !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32])>, PhysicalStorageBuffer>
        %addr_p = spirv.AccessChain %desc[%c1] : !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32])>, PhysicalStorageBuffer>, i32 -> !spirv.ptr<i64, PhysicalStorageBuffer>
        %addr = spirv.Load "PhysicalStorageBuffer" %addr_p ["Aligned", 8] : i64
        %data = spirv.ConvertUToPtr %addr : i64 to !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>
        // Index by the global invocation id (contiguous buffer).
        %gid_a = spirv.mlir.addressof @gid : !spirv.ptr<vector<3xi32>, Input>
        %gid_v = spirv.Load "Input" %gid_a : vector<3xi32>
        %tid = spirv.CompositeExtract %gid_v[0 : i32] : vector<3xi32>
        %elem = spirv.AccessChain %data[%c0, %tid] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>, i32, i32 -> !spirv.ptr<f32, PhysicalStorageBuffer>
        %v = spirv.Load "PhysicalStorageBuffer" %elem ["Aligned", 4] : f32
        %two = spirv.Constant 2.000000e+00 : f32
        %r = spirv.FMul %v, %two : f32
        spirv.Store "PhysicalStorageBuffer" %elem, %r ["Aligned", 4] : f32
        spirv.Return
      }
      spirv.EntryPoint "GLCompute" @double_buffer, @gid, @pc
      spirv.ExecutionMode @double_buffer "LocalSize", 1, 1, 1
    }
  }

  func.func @main() {
    %buffer = memref.alloc() : memref<8xf32>
    %value = arith.constant 1.5 : f32
    %buffer_casted = memref.cast %buffer : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%buffer_casted, %value) : (memref<?xf32>, f32) -> ()

    %cst1 = arith.constant 1 : index
    %cst8 = arith.constant 8 : index
    gpu.launch_func @kernels::@double_buffer
        blocks in (%cst8, %cst1, %cst1) threads in (%cst1, %cst1, %cst1)
        args(%buffer : memref<8xf32>)

    %buffer_out = memref.cast %buffer : memref<8xf32> to memref<*xf32>
    call @printMemrefF32(%buffer_out) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
