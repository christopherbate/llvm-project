add_mlir_dialect_library(MLIRFuncDialect
  FuncOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func

  DEPENDS
  MLIRFuncOpsIncGen

  # FuncOps.cpp includes generated Bufferization interfaces.
  LINK_LIBS
  PRIVATE
  MLIRBufferizableOpInterfaceIncGen
  MLIRConvertToEmitCPatternInterfaceIncGen
  MLIRConvertToLLVMInterfaceIncGen
  MLIRTransformsDialectInterfaceIncGen
  PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
  )
