# LLVM MLIR Example

## What is this?

LLVM MLIRの学習のために作成したサンプルプログラム。

## Build

任意の場所に[LLVM](https://github.com/llvm/llvm-project)をクローンし、cmakeを実行してください。

```sh
git clone https://github.com/llvm/llvm-project.git
mkdir llvm-project/build
cd llvm-project/build
# MSVC (Visual Studio 17 2022)を使う場合
cmake ../llvm -G "Visual Studio 17 2022" -A x64 -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_TARGETS_TO_BUILD="Native" -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_ENABLE_DIA_SDK=OFF
cmake --build . --target MLIRSupport MLIRIR MLIRParser MLIRDialect MLIRTransforms MLIRPass MLIRTargetLLVMIRExport MLIRLLVMToLLVMIRTranslation MLIRBuiltinToLLVMIRTranslation MLIRLLVMCommonConversion MLIRFuncDialect MLIRFuncToLLVM --config Release
```

各サンプルプログラムのREADME.mdを参照して、各サンプルプログラムのビルドに必要なLLVMのライブラリをビルドしてください。

次の環境変数を定義してください。

- `LLVM_LLVM_INCLUDE_PATH`: `llvm-project/llvm/include/`
- `LLVM_MLIR_INCLUDE_PATH`: `llvm-project/mlir/include/`
- `LLVM_LLVM_BUILD_INCLUDE_PATH`
  - MSVC: `llvm-project/build/include/`
- `LLVM_MLIR_BUILD_INCLUDE_PATH`
  - MSVC: `llvm-project/build/tools/mlir/include/`
- `LLVM_LIBRARY_PATH`
  - MSVC: `llvm-project/build/Release/lib/`

利用するコンパイラに合わせて、次のPythonスクリプトを実行してください。
ただし、サンプルプログラムの番号(01-minimalであれば`01`)をコマンドライン引数として与えてください。

- MSVC: `build-with-msvc.py`
