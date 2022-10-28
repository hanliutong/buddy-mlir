add_mlir_dialect_library(BuddyRVK
  RVKDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}/include/Dialect/RISCVK

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  )
