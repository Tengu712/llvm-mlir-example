#include "SampleDialect.h"
#include "SamplePasses.h"

#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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
  llvm::cl::ParseCommandLineOptions(argc, argv, "05 sample\n");

  // 入力ファイルを開く
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return 1;
  }

  // MLIRContextの初期化
  // 自作したSampleDialectを追加
  mlir::MLIRContext context;
  context.getOrLoadDialect<sample::SampleDialect>();
  context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  // パース
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file: " << inputFilename << "\n";
    return 1;
  }

  // 自作したLoweringパスを適用
  {
    mlir::PassManager passManager(&context);
    passManager.addPass(sample::createLowerToLLVM());
    if (mlir::failed(passManager.run(*module))) {
      llvm::errs() << "Failed to run loop fusion pass.\n";
      return 1;
    }
  }

  // LLVMDialectへLowering
  mlir::ConversionTarget target(context);
  target.addLegalOp<mlir::ModuleOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::func::FuncDialect>();
  mlir::LLVMTypeConverter typeConverter(&context);
  mlir::RewritePatternSet patterns(&context);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
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
