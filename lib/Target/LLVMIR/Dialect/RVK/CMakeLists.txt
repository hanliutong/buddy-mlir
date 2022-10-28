add_mlir_translation_library(BuddyRVKToLLVMIRTranslation
  RVKToLLVMIRTranslation.cpp

  DEPENDS
  BuddyRVKConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  BuddyRVK
  )
