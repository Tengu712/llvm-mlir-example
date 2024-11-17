# 02 Noop

## About

Func Dialectを使えるMLIRをパースしてダンプするだけのサンプルプログラム。

## Build

次のように、必要なLLVMのライブラリをビルド。

```
$ cmake --build . --target MLIRSupport MLIRParser MLIRFuncDialect --config Release
```

## Result

```
$ 02-noop emp-func.mlir
module {
  func.func @test() {
    return
  }
}
```

## Memo

- `MLIRSupport`: `llvm::cl::ParseCommandLineOptions()`
- `MLIRParser`: `llvm::cl::parseSourceFile()`
- `MLIRFuncDialect`: `mlir::func::FuncDialect`
