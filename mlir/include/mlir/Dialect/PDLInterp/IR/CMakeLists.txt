add_mlir_dialect(PDLInterpOps pdl_interp)
# PDLInterp.h includes generated PDL operation and type declarations.
target_link_libraries(MLIRPDLInterpOpsIncGen INTERFACE MLIRPDLOpsIncGen)
add_mlir_doc(PDLInterpOps PDLInterpOps Dialects/ -gen-dialect-doc -dialect=pdl_interp)
