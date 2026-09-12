set(LLVM_OPTIONAL_SOURCES
  ArithOps.cpp
  ArithDialect.cpp
  InferIntDivisibilityOpInterfaceImpl.cpp
  InferIntRangeInterfaceImpls.cpp
  ValueBoundsOpInterfaceImpl.cpp
  )

set(LLVM_TARGET_DEFINITIONS ArithCanonicalization.td)
mlir_tablegen(ArithCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRArithCanonicalizationIncGen)

add_mlir_dialect_library(MLIRArithDialect
  ArithOps.cpp
  ArithDialect.cpp
  InferIntDivisibilityOpInterfaceImpl.cpp
  InferIntRangeInterfaceImpls.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith

  DEPENDS
  MLIRArithOpsIncGen
  MLIRArithOpsInterfacesIncGen
  # TODO: Give this generated interface a standalone logical library target.
  MLIRConvertToEmitCPatternInterfaceIncGen
  MLIRConvertToLLVMInterfaceIncGen
  MLIRTransformsDialectInterfaceIncGen

  # The sources include generated interface definitions.
  LINK_LIBS
  PRIVATE
  MLIRBufferDeallocationOpInterfaceIncGen
  MLIRBufferizableOpInterfaceIncGen
  MLIRDestinationStyleOpInterfaceIncGen
  MLIRValueBoundsOpInterfaceIncGen
  PUBLIC
  MLIRControlFlowInterfacesIncGen
  MLIRCastInterfaces
  MLIRDialect
  MLIRInferIntDivisibilityOpInterface
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRUBDialect
  MLIRVectorInterfaces
  )

add_mlir_dialect_library(MLIRArithValueBoundsOpInterfaceImpl
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arith

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRValueBoundsOpInterface
  )
