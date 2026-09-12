set(LLVM_OPTIONAL_SOURCES ByteCode.cpp)

set(pdl_header_targets)
if(MLIR_ENABLE_PDL_IN_PATTERNMATCH)
  set(pdl_header_targets
    MLIRConversionPassIncGen
    MLIRPDLInterpOpsIncGen
    MLIRPDLOpsIncGen)
endif()

add_mlir_library(MLIRRewrite
  FrozenRewritePatternSet.cpp
  PatternApplicator.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite

  # FrozenRewritePatternSet.cpp includes PDLInterp and PDL when enabled.
  LINK_LIBS
  PRIVATE
  ${pdl_header_targets}
  PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )

if(MLIR_ENABLE_PDL_IN_PATTERNMATCH)
  add_mlir_library(MLIRRewritePDL
    ByteCode.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Rewrite

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPDLToPDLInterp
    MLIRSideEffectInterfaces
  )

  target_link_libraries(MLIRRewrite PUBLIC
    MLIRPDLDialect
    MLIRPDLInterpDialect
    MLIRPDLToPDLInterp
    MLIRRewritePDL)
endif()
