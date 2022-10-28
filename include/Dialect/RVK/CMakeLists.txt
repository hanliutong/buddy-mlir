add_mlir_dialect(RVK rvk)
add_mlir_doc(RVK RVK Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS RVK.td)
mlir_tablegen(RVKConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(BuddyRVKConversionsIncGen)
