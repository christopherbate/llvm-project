add_mlir_dialect_library(MLIRSCFDialect
  SCF.cpp
  DeviceMappingInterface.cpp
  MemorySlot.cpp
  ValueBoundsOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/SCF

  DEPENDS
  MLIRSCFOpsIncGen
  MLIRDeviceMappingInterfacesIncGen

  LINK_LIBS
  PRIVATE
  MLIRAlignmentAttrInterfaceIncGen
  MLIRBufferDeallocationOpInterfaceIncGen
  MLIRBufferizableOpInterfaceIncGen
  PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLoopLikeInterface
  MLIRMemorySlotUtils
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRValueBoundsOpInterface
  MLIRTransformUtils
  )
