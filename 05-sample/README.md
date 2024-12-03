# 05 sample

## About

自作のDialectを定義して定数を返すだけの関数を定義するサンプルプログラム。

## Build

次のように、必要なLLVMのライブラリをビルド。

```
$ cmake --build . --target MLIRSupport MLIRParser MLIRFuncDialect MLIRFuncToLLVM MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation --config Release
```

## Result

```
$ 05-sample sample-constant.mlir
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define i32 @ret_sample_contant() {
  ret i32 13
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
