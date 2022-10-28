add_mlir_dialect_library(BuddyRVKTransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  BuddyRVKConversionsIncGen

  LINK_LIBS PUBLIC
  BuddyRVK
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
