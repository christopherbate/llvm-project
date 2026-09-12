add_mlir_conversion_library(MLIRSPIRVAttrToLLVMConversion
  AttrToLLVMConverter.cpp

  # AttrToLLVMConverter.h includes SPIRVEnums.h, which includes generated
  # builtin attribute declarations.
  LINK_LIBS PUBLIC
  MLIRBuiltinAttributeInterfacesIncGen
  MLIRBuiltinAttributesIncGen
  MLIRBuiltinLocationAttributesIncGen
  MLIRBuiltinTypeInterfacesIncGen
  MLIROpAsmInterfaceIncGen
  MLIRSPIRVEnumsIncGen
)
