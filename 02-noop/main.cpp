#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

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
  // - FuncDialect
  mlir::MLIRContext context;
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);

  // パース
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input file: " << inputFilename << "\n";
    return 1;
  }

  // パース結果を表示
  module->dump();

  return 0;
}
