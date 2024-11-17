#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <string>

static llvm::cl::opt<std::string> inputFilename(
  llvm::cl::Positional,
  llvm::cl::desc("<file-path>"),
  llvm::cl::init("-")
);

int main(int argc, char **argv) {
  mlir::registerMLIRContextCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "02 noop\n");

  // 入力ファイルを開く
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }

  // MLIRContextの初期化
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();

  // パース
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file: " << inputFilename << "\n";
    return 1;
  }

  llvm::outs() << "==================== BEFORE ====================\n";
  module->dump();
  llvm::outs() << "==================== END =======================\n";

  // Affine DialectのLoop Fusion Passを適用
  // NOTE: dumpしたいからわざわざ分けているだけなので、
  //       普通なら次のPass適用にくっつける。
  {
    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::affine::createLoopFusionPass());
    if (mlir::failed(passManager.run(*module))) {
      llvm::errs() << "Failed to run loop fusion pass.\n";
      return 1;
    }
  }

  llvm::outs() << "==================== AFTER ====================\n";
  module->dump();
  llvm::outs() << "==================== END =======================\n";

  // AffineをControlFlowに変換するパスを適用
  // - affine -> scf -> cf
  {
    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::createLowerAffinePass());
    passManager.addPass(mlir::createConvertSCFToCFPass());
    if (mlir::failed(passManager.run(*module))) {
      llvm::errs() << "Failed to run passes to convert affine to cf.\n";
      return 1;
    }
  }

  // LLVMDialectへLowering
  mlir::ConversionTarget target(context);
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  mlir::LLVMTypeConverter typeConverter(&context);
  mlir::RewritePatternSet patterns(&context);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  if (mlir::failed(mlir::applyFullConversion(*module, target, std::move(patterns)))) {
    llvm::errs() << "Failed to apply full conversion.\n";
    return 1;
  }

  // LLVM IRへ変換
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module.get(), llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate to LLVM IR.\n";
    return 1;
  }

  // LLVM IRを出力
  std::string irBuffer;
  llvm::raw_string_ostream irStream(irBuffer);
  llvmModule->print(irStream, nullptr);
  llvm::outs() << irBuffer;

  return 0;
}
