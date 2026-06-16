// Multi-buffer variant of physical-storage-buffer.mlir: the kernel receives the
// device addresses of three buffers through a single argument buffer (one
// device-resident MLIR ranked memref descriptor of {allocatedPtr, alignedPtr,
// offset, size, stride} per buffer, in binding/argument order) whose device
// address is the sole push constant, and computes out[i] = a[i] + b[i].

// RUN: mlir-opt %s -test-vulkan-runner-pipeline='kernel-is-spirv use-buffer-device-address' \
// RUN:   | mlir-runner - --shared-libs=%mlir_vulkan_runtime,%mlir_runner_utils --entry-point-result=void | FileCheck %s

// CHECK: [3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3]
module attributes {gpu.container_module} {
  gpu.module @kernels [#spirv.target_env<
      #spirv.vce<v1.5, [Shader, Int64, PhysicalStorageBufferAddresses],
                 [SPV_KHR_physical_storage_buffer]>,
      #spirv.resource_limits<>>] {
    gpu.func @vector_add(%arg0 : memref<8xf32>, %arg1 : memref<8xf32>,
                         %arg2 : memref<8xf32>) kernel {
      gpu.return
    }

    spirv.module PhysicalStorageBuffer64 GLSL450 requires #spirv.vce<v1.5,
        [Shader, Int64, PhysicalStorageBufferAddresses],
        [SPV_KHR_physical_storage_buffer]> {
      spirv.GlobalVariable @pc : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>
      spirv.GlobalVariable @gid built_in("GlobalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
      spirv.func @vector_add() "None" {
        %c0 = spirv.Constant 0 : i32
        // The alignedPtr field of descriptor k: each ranked descriptor occupies
        // 5 i64 slots and alignedPtr is field 1, so they are at 1, 6, 11.
        %ca = spirv.Constant 1 : i32
        %cb = spirv.Constant 6 : i32
        %cc = spirv.Constant 11 : i32
        // Root pointer (device address of the argument buffer).
        %pc = spirv.mlir.addressof @pc : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>
        %root_ptr = spirv.AccessChain %pc[%c0] : !spirv.ptr<!spirv.struct<(i64 [0])>, PushConstant>, i32 -> !spirv.ptr<i64, PushConstant>
        %root = spirv.Load "PushConstant" %root_ptr : i64
        // Argument buffer holds three contiguous ranked descriptors
        // {allocatedPtr, alignedPtr, offset, size, stride} (15 i64 fields).
        %args = spirv.ConvertUToPtr %root : i64 to !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32], i64 [40], i64 [48], i64 [56], i64 [64], i64 [72], i64 [80], i64 [88], i64 [96], i64 [104], i64 [112])>, PhysicalStorageBuffer>
        %a_addr_p = spirv.AccessChain %args[%ca] : !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32], i64 [40], i64 [48], i64 [56], i64 [64], i64 [72], i64 [80], i64 [88], i64 [96], i64 [104], i64 [112])>, PhysicalStorageBuffer>, i32 -> !spirv.ptr<i64, PhysicalStorageBuffer>
        %a_addr = spirv.Load "PhysicalStorageBuffer" %a_addr_p ["Aligned", 8] : i64
        %b_addr_p = spirv.AccessChain %args[%cb] : !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32], i64 [40], i64 [48], i64 [56], i64 [64], i64 [72], i64 [80], i64 [88], i64 [96], i64 [104], i64 [112])>, PhysicalStorageBuffer>, i32 -> !spirv.ptr<i64, PhysicalStorageBuffer>
        %b_addr = spirv.Load "PhysicalStorageBuffer" %b_addr_p ["Aligned", 8] : i64
        %out_addr_p = spirv.AccessChain %args[%cc] : !spirv.ptr<!spirv.struct<(i64 [0], i64 [8], i64 [16], i64 [24], i64 [32], i64 [40], i64 [48], i64 [56], i64 [64], i64 [72], i64 [80], i64 [88], i64 [96], i64 [104], i64 [112])>, PhysicalStorageBuffer>, i32 -> !spirv.ptr<i64, PhysicalStorageBuffer>
        %out_addr = spirv.Load "PhysicalStorageBuffer" %out_addr_p ["Aligned", 8] : i64

        %a_ptr = spirv.ConvertUToPtr %a_addr : i64 to !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>
        %b_ptr = spirv.ConvertUToPtr %b_addr : i64 to !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>
        %out_ptr = spirv.ConvertUToPtr %out_addr : i64 to !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>

        %gid_a = spirv.mlir.addressof @gid : !spirv.ptr<vector<3xi32>, Input>
        %gid_v = spirv.Load "Input" %gid_a : vector<3xi32>
        %tid = spirv.CompositeExtract %gid_v[0 : i32] : vector<3xi32>

        %a_elem = spirv.AccessChain %a_ptr[%c0, %tid] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>, i32, i32 -> !spirv.ptr<f32, PhysicalStorageBuffer>
        %b_elem = spirv.AccessChain %b_ptr[%c0, %tid] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>, i32, i32 -> !spirv.ptr<f32, PhysicalStorageBuffer>
        %out_elem = spirv.AccessChain %out_ptr[%c0, %tid] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, PhysicalStorageBuffer>, i32, i32 -> !spirv.ptr<f32, PhysicalStorageBuffer>

        %a_val = spirv.Load "PhysicalStorageBuffer" %a_elem ["Aligned", 4] : f32
        %b_val = spirv.Load "PhysicalStorageBuffer" %b_elem ["Aligned", 4] : f32
        %sum = spirv.FAdd %a_val, %b_val : f32
        spirv.Store "PhysicalStorageBuffer" %out_elem, %sum ["Aligned", 4] : f32
        spirv.Return
      }
      spirv.EntryPoint "GLCompute" @vector_add, @gid, @pc
      spirv.ExecutionMode @vector_add "LocalSize", 1, 1, 1
    }
  }

  func.func @main() {
    %a = memref.alloc() : memref<8xf32>
    %b = memref.alloc() : memref<8xf32>
    %out = memref.alloc() : memref<8xf32>
    %v1 = arith.constant 1.1 : f32
    %v2 = arith.constant 2.2 : f32
    %v0 = arith.constant 0.0 : f32
    %a_casted = memref.cast %a : memref<8xf32> to memref<?xf32>
    %b_casted = memref.cast %b : memref<8xf32> to memref<?xf32>
    %out_casted = memref.cast %out : memref<8xf32> to memref<?xf32>
    call @fillResource1DFloat(%a_casted, %v1) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%b_casted, %v2) : (memref<?xf32>, f32) -> ()
    call @fillResource1DFloat(%out_casted, %v0) : (memref<?xf32>, f32) -> ()

    %cst1 = arith.constant 1 : index
    %cst8 = arith.constant 8 : index
    gpu.launch_func @kernels::@vector_add
        blocks in (%cst8, %cst1, %cst1) threads in (%cst1, %cst1, %cst1)
        args(%a : memref<8xf32>, %b : memref<8xf32>, %out : memref<8xf32>)

    %out_unranked = memref.cast %out : memref<8xf32> to memref<*xf32>
    call @printMemrefF32(%out_unranked) : (memref<*xf32>) -> ()
    return
  }
  func.func private @fillResource1DFloat(%0 : memref<?xf32>, %1 : f32)
  func.func private @printMemrefF32(%ptr : memref<*xf32>)
}
