add_mlir_library(MLIRDialectUtils
  IndexingUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp
  VerificationUtils.cpp

  DEPENDS
  MLIRDialectUtilsIncGen
  MLIRArithOpsIncGen

  # ReshapeOpsUtils.h includes Arith.h, but this utility library does not link
  # the Arith library itself.
  LINK_LIBS PUBLIC
  MLIRArithOpsInterfacesIncGen
  MLIRControlFlowInterfacesIncGen
  MLIRInferIntDivisibilityOpInterfaceIncGen
  MLIRInferTypeOpInterfaceIncGen
  MLIRVectorInterfacesIncGen
  MLIRIR
)
