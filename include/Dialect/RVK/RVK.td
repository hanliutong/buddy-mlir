#ifndef RVK_OPS
#define RVK_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"


//===----------------------------------------------------------------------===//
// RVV dialect definition
//===----------------------------------------------------------------------===//

def RVK_Dialect : Dialect {
  let name = "rvk";
  let cppNamespace = "::buddy::rvk";
  let summary = "Basic dialect to target RISC-V Cryptography Extension";
  let description = [{
    For more details about RVK, please see the 
    [RVK specification](https://github.com/riscv/riscv-crypto).
  }];
}

//===----------------------------------------------------------------------===//
// RVV intrinsic operation definitions
//===----------------------------------------------------------------------===//

class RVK_AES64ESM_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVK_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[], 
                  /*list<int> overloadedOperands=*/[], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/1>;


def RVKAES64ESMOP : RVK_AES64ESM_IntrOp<"aes64esm">,
  Arguments<(ins AnyInteger, AnyInteger)>;


#endif // RVV_OPS
