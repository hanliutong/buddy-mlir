//===- LegalizeForLLVMExport.cpp - Prepare RVK for LLVM translation -------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RVK/RVKDialect.h"
#include "RVK/Transforms.h"

using namespace mlir;
using namespace buddy::rvk;

template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");

    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};


void mlir::configureRVKLegalizeForExportTarget(LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<RVKAES64ESMOP>();
  // clang-format on
}
