# 03 To LLVM IR

## About

Func Dialectを使えるMLIRをLLVM IRに変換するだけのサンプルプログラム。

## Build

次のように、必要なLLVMのライブラリをビルド。

```
$ cmake --build . --target MLIRSupport MLIRParser MLIRFuncDialect MLIRFuncToLLVM MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation --config Release
```

## Result

```
$ 03-to-llvmir emp-func.mlir
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @test() {
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
```

## Memo

- `MLIRSupport`: `llvm::cl::ParseCommandLineOptions()`
- `MLIRParser`: `llvm::cl::parseSourceFile()`
- `MLIRFuncDialect`: `mlir::func::FuncDialect`
- `MLIRFuncToLLVM`: `mlir::populateFuncToLLVMConversionPatterns()`
- `MLIRBuiltinToLLVMIRTranslation`: `mlir::registerBuiltinDialectTranslation()`
- `MLIRLLVMToLLVMIRTranslation`: `mlir::registerLLVMDialectTranslation()`
