add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name NvGpu)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix NvGpu)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix NvGpu)
add_public_tablegen_target(MLIRNvGpuPassIncGen)

add_mlir_doc(Passes NvGpuPasses ./ -gen-pass-doc)

