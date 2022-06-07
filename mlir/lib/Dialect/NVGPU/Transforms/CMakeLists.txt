add_mlir_dialect_library(MLIRNvGpuTransforms
  OptimizeSharedMemory.cpp  

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Linalg/NvGpu

  DEPENDS
  MLIRNvGpuPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRFunc
  MLIRIR
  MLIRMemRef
  MLIRPass  
  MLIRTensor    
  MLIRTransforms
  MLIRTransformUtils
  MLIRVector
  MLIRVectorTransforms
  MLIRVectorUtils  
)
