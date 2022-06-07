// RUN: mlir-opt %s -split-input-file --pass-pipeline='func.func(nvgpu-optimize-shared-memory)' | FileCheck %s

// CHECK: @optimize_128x32xf16([[arg0:%.+]]: memref<{{.*}}>, [[ldRow:%.+]]: index, [[ldCol:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index)
func.func @optimize_128x32xf16(%arg0: memref<128x128xf16>,
                               %ldRow: index, %ldCol: index,
                               %stRow: index, %stCol: index,
                               %fragRow: index, %fragCol :index)
                                -> vector<4x2xf16> {
  %shm = memref.alloc() : memref<128x32xf16, 3>
  // CHECK: [[shm:%.+]] = memref.alloc
  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[src_bits:%.+]] = arith.andi [[stRow]], [[c6]]
  // CHECK: [[c2:%.+]] = arith.constant 2 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[src_bits]], [[c2]]
  // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol]], [[xorBits]]  
  
  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shm]][[[stRow]], [[stColPerm]]]
  %0 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shm[%stRow, %stCol], 8
      : memref<128x128xf16> to memref<128x32xf16, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c2:%.+]] = arith.constant 2 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c2]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: nvgpu.ldmatrix [[shm]][[[fragRow]], [[fragColPerm]]]
  %mat = nvgpu.ldmatrix %shm[%fragRow, %fragCol] {numTiles = 4 : i32, transpose = false}
      : memref<128x32xf16, 3> -> vector<4x2xf16>
  return %mat: vector<4x2xf16>
}

// -----

// CHECK: @optimize_32x128xf16([[arg0:%.+]]: memref<{{.*}}>, [[ldRow:%.+]]: index, [[ldCol:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index)
func.func @optimize_32x128xf16(%arg0: memref<128x128xf16>,
                               %ldRow: index, %ldCol: index,
                               %stRow: index, %stCol: index,
                               %fragRow: index, %fragCol :index)
                                -> vector<4x2xf16> {
  %shm = memref.alloc() : memref<32x128xf16, 3>
  // CHECK: [[shm:%.+]] = memref.alloc
  // CHECK: [[c15:%.+]] = arith.constant 15 : index
  // CHECK: [[src_bits:%.+]] = arith.andi [[stRow]], [[c15]]
  // CHECK: [[c3:%.+]] = arith.constant 3 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[src_bits]], [[c3]]
  // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol]], [[xorBits]]  
  
  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shm]][[[stRow]], [[stColPerm]]]
  %0 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shm[%stRow, %stCol], 8
      : memref<128x128xf16> to memref<32x128xf16, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: [[c15:%.+]] = arith.constant 15 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c15]]
  // CHECK: [[c3:%.+]] = arith.constant 3 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c3]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: nvgpu.ldmatrix [[shm]][[[fragRow]], [[fragColPerm]]]      
  %mat = nvgpu.ldmatrix %shm[%fragRow, %fragCol] {numTiles = 4 : i32, transpose = false}
      : memref<32x128xf16, 3> -> vector<4x2xf16>
  return %mat: vector<4x2xf16>
}

// -----

// CHECK: @optimize_64x16xf32([[arg0:%.+]]: memref<{{.*}}>, [[ldRow:%.+]]: index, [[ldCol:%.+]]: index, [[stRow:%.+]]: index, [[stCol:%.+]]: index, [[fragRow:%.+]]: index, [[fragCol:%.+]]: index)
func.func @optimize_64x16xf32 (%arg0: memref<128x128xf32>,
                               %ldRow: index, %ldCol: index,
                               %stRow: index, %stCol: index,
                               %fragRow: index, %fragCol :index)
                                -> (vector<4x1xf32>, f32) {
  %shm = memref.alloc() : memref<64x16xf32, 3>
  // CHECK: [[shm:%.+]] = memref.alloc
  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[src_bits:%.+]] = arith.andi [[stRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[src_bits]], [[c1]]
  // CHECK: [[stColPerm:%.+]] = arith.xori [[stCol]], [[xorBits]]  
  
  // CHECK: nvgpu.device_async_copy [[arg0]][[[ldRow]], [[ldCol]]], [[shm]][[[stRow]], [[stColPerm]]]
  %0 = nvgpu.device_async_copy %arg0[%ldRow, %ldCol], %shm[%stRow, %stCol], 8
      : memref<128x128xf32> to memref<64x16xf32, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32}

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c1]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: nvgpu.ldmatrix [[shm]][[[fragRow]], [[fragColPerm]]]  
  %mat = nvgpu.ldmatrix %shm[%fragRow, %fragCol] {numTiles = 4 : i32, transpose = false}
      : memref<64x16xf32, 3> -> vector<4x1xf32>

  // CHECK: [[c6:%.+]] = arith.constant 6 : index
  // CHECK: [[srcBits:%.+]] = arith.andi [[fragRow]], [[c6]]
  // CHECK: [[c1:%.+]] = arith.constant 1 : index
  // CHECK: [[xorBits:%.+]] = arith.shli [[srcBits]], [[c1]]
  // CHECK: [[fragColPerm:%.+]] = arith.xori [[fragCol]], [[xorBits]]
  // CHECK: memref.load [[shm]][[[fragRow]], [[fragColPerm]]]
  %elem = memref.load %shm[%fragRow, %fragCol] : memref<64x16xf32, 3>
  return %mat, %elem: vector<4x1xf32>, f32
}