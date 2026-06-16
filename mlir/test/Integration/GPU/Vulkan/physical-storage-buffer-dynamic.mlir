// Exercises the buffer-device-address path with a *dynamically shaped* memref.
// The kernel receives a device-resident MLIR ranked memref descriptor
// ({allocatedPtr, alignedPtr, offset, size, stride}, all i64) through an
// argument buffer whose device address is the sole push constant; it
// reconstructs the element address with offset + tid*stride. The runtime stays
// type-agnostic: the compiler hands it the memref via the standard unranked
// descriptor convention and the launch selects this path via a marker.

// RUN: mlir-opt %s -test-vulkan-runner-pipeline='kernel-is-spirv use-buffer-device-address' \
// RUN:   | mlir-runner - --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils --entry-point-result=void | FileCheck %s

// CHECK: [3, 3, 3, 3, 3, 3, 3, 3]
module attributes {gpu.container_module} {
  gpu.module @kernels [#spirv.target_env<
      #spirv.vce<v1.5, [Shader, Int64, PhysicalStorageBufferAddresses],
                 [SPV_KHR_physical_storage_buffer]>,
      #spirv.resource_limits<>>] {
    gpu.func @double_dyn(%arg0 : memref<?xf32>) kernel {
      gpu.return
    }

    spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.5,
        [Shader, Int64, PhysicalStorageBufferAddresses],
        [SPV_KHR_physical_storage_buffer]> {
      spirv.GlobalVariable @pc : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>
      spirv.GlobalVariable @gid built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
      spirv.func @double_dyn() "None" {
        %c0 = spirv.Constant 0 : i32
        %c1 = spirv.Constant 1 : i32
        %c2 = spirv.Constant 2 : i32
        %c4 = spirv.Constant 4 : i32
        // Root pointer (device address of the argument buffer) from the push constant.
        %pc = spirv.mlir.addressof @pc : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>
        %root_ptr = spirv.AccessChain %pc[%c0] : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>, i32 -> !spirv.ptr<i64, PushConstant>
        %root = spirv.Load "PushConstant" %root_ptr : i64
        // Ranked descriptor: { allocatedPtr, alignedPtr, offset, size, stride }.
        %desc = spirv.ConvertUToPtr %root : i64 to !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32])>, PhysicalStorageBuffer>
        %addr_p = spirv.AccessChain %desc[%c1] : !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32])>, PhysicalStorageBuffer>, i32 -> !spirv.ptr<i64, PhysicalStorageBuffer>
        %addr = spirv.Load "PhysicalStorageBuffer" %addr_p ["Aligned", 8] : i64
        %off_p = spirv.AccessChain %desc[%c2] : !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32])>, PhysicalStorageBuffer>, i32 -> !spirv.ptr<i64, PhysicalStorageBuffer>
        %off = spirv.Load "PhysicalStorageBuffer" %off_p ["Aligned", 8] : i64
        %str_p = spirv.AccessChain %desc[%c4] : !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32])>, PhysicalStorageBuffer>, i32 -> !spirv.ptr<i64, PhysicalStorageBuffer>
        %str = spirv.Load "PhysicalStorageBuffer" %str_p ["Aligned", 8] : i64
        // Data pointer (alignedPtr holds the buffer's device address).
        %data = spirv.ConvertUToPtr %addr : i64 to !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>
        // index = offset + tid * stride
        %gid_a = spirv.mlir.addressof @gid : !spirv.ptr<vector<3xi32>, Input>
        %gid_v = spirv.Load "Input" %gid_a : vector<3xi32>
        %tid32 = spirv.CompositeExtract %gid_v[0 : i32] : vector<3xi32>
        %tid = spirv.UConvert %tid32 : i32 to i64
        %mul = spirv.IMul %tid, %str : i64
        %idx = spirv.IAdd %off, %mul : i64
        %elem = spirv.AccessChain %data[%c0, %idx] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>, i32, i64 -> !spirv.ptr<f32, PhysicalStorageBuffer>
        %v = spirv.Load "PhysicalStorageBuffer" %elem ["Aligned", 4] : f32
        %two = spirv.Constant 2.000000e+00 : f32
        %r = spirv.FMul %v, %two : f32
        spirv.Store "PhysicalStorageBuffer" %elem, %r ["Aligned", 4] : f32
        spirv.Return
      }
      spirv.EntryPoint "GLCompute" @double_dyn, @gid, @pc
      spirv.ExecutionMode @double_dyn "LocalSize", 1, 1, 1
    }
  }

  func.func @main() {
    %c8 = arith.constant 8 : index
    %buffer = memref.alloc(%c8) : memref<?xf32>
    %value = arith.constant 1.5 : f32
    call @fillResource1DFloat(%buffer, %value) : (memref<?xf32>, f32) -> ()

    %cst1 = arith.constant 1 : index
    gpu.launch_func @kernels::@double_dyn
        blocks in (%c8, %cst1, %cst1) threads in (%cst1, %cst1, %cst1)
        args(%buffer : memref<?xf32>)

    %buffer_out = memref.cast %buffer : memref<?xf32> to memref<*xf32>
    call @printMemrefF32(%buffer_out) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
