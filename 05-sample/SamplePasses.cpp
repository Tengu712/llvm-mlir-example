#include "SamplePasses.h"

#include "SampleOps.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
  class ConstantOpLowering: public mlir::ConversionPattern {
  public:
    ConstantOpLowering(mlir::MLIRContext *ctx): ConversionPattern("sample.constant", 1, ctx) {}

    mlir::LogicalResult matchAndRewrite(mlir::Operation *op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter &rewriter) const final {
      auto constantOp = llvm::cast<sample::ConstantOp>(op);

      auto llvmType = rewriter.getI32Type();
      auto llvmValue = rewriter.create<mlir::LLVM::ConstantOp>(op->getLoc(), llvmType, rewriter.getI32IntegerAttr(constantOp.getValue()));

      rewriter.replaceOp(op, llvmValue);
      return mlir::success();
    }
  };

  class SampleToLLVMLowerPass: public mlir::PassWrapper<SampleToLLVMLowerPass, mlir::OperationPass<mlir::ModuleOp>> {
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SampleToLLVMLowerPass)

    void runOnOperation() final {
      mlir::ConversionTarget target(getContext());
      target.addLegalDialect<mlir::LLVM::LLVMDialect>();

      mlir::RewritePatternSet patterns(&getContext());
      patterns.add<ConstantOpLowering>(&getContext());

      if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
      }
    }
  };
}

std::unique_ptr<mlir::Pass> sample::createLowerToLLVM() {
  return std::make_unique<SampleToLLVMLowerPass>();
}
