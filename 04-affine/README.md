# 04 Affine

## About

Affine DialectのLoop Fusion Passを使って最適化を行うサンプルプログラム。

## Build

次のように、必要なLLVMのライブラリをビルド。

```
$ cmake --build . --target MLIRSupport MLIRParser MLIRFuncDialect MLIRFuncToLLVM MLIRBuiltinToLLVMIRTranslation MLIRLLVMToLLVMIRTranslation MLIRAffineDialect MLIRAffineToStandard MLIRSCFDialect MLIRMemRefToLLVM --config Release
```

## Result

```
$ 04-affine affine-loop.mlir
==================== BEFORE ====================
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<100xf32>
    %alloc_0 = memref.alloc() : memref<100xf32>
    affine.for %arg0 = 0 to 100 {
      %0 = affine.load %alloc[%arg0] : memref<100xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %alloc[%arg0] : memref<100xf32>
    }
    affine.for %arg0 = 0 to 100 {
      %0 = affine.load %alloc[%arg0] : memref<100xf32>
      %1 = arith.addf %0, %0 : f32
      affine.store %1, %alloc_0[%arg0] : memref<100xf32>
    }
    return
  }
}
==================== END =======================
==================== AFTER ====================
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<1xf32>
    %alloc_0 = memref.alloc() : memref<100xf32>
    affine.for %arg0 = 0 to 100 {
      %0 = affine.load %alloc[0] : memref<1xf32>
      %1 = arith.mulf %0, %0 : f32
      affine.store %1, %alloc[0] : memref<1xf32>
      %2 = affine.load %alloc[0] : memref<1xf32>
      %3 = arith.addf %2, %2 : f32
      affine.store %3, %alloc_0[%arg0] : memref<100xf32>
    }
    return
  }
}
==================== END =======================
; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define void @main() {
  %1 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i64 1) to i64))
  %2 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %1, 0
  %3 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %2, ptr %1, 1
  %4 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %3, i64 0, 2
  %5 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, i64 1, 3, 0
  %6 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, i64 1, 4, 0
  %7 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i64 100) to i64))
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %7, 0
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, ptr %7, 1
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 0, 2
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 100, 3, 0
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, i64 1, 4, 0
  br label %13

13:                                               ; preds = %16, %0
  %14 = phi i64 [ %29, %16 ], [ 0, %0 ]
  %15 = icmp slt i64 %14, 100
  br i1 %15, label %16, label %30

16:                                               ; preds = %13
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %18 = getelementptr float, ptr %17, i64 0
  %19 = load float, ptr %18, align 4
  %20 = fmul float %19, %19
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %22 = getelementptr float, ptr %21, i64 0
  store float %20, ptr %22, align 4
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %6, 1
  %24 = getelementptr float, ptr %23, i64 0
  %25 = load float, ptr %24, align 4
  %26 = fadd float %25, %25
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, 1
  %28 = getelementptr float, ptr %27, i64 %14
  store float %26, ptr %28, align 4
  %29 = add i64 %14, 1
  br label %13

30:                                               ; preds = %13
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
- `MLIRAffineDialect`: `mlir::affine::AffineDialect`
- `MLIRAffineToStandard`: `mlir::createLowerAffinePass()`
- `MLIRSCFDialect`: `mlir::scf::SCFDialect`
- `MLIRSCFToControlFlow`: `mlir::createConvertSCFToCFPass()`
- `MLIRMemRefToLLVM`: `mlir::populateFinalizeMemRefToLLVMConversionPatterns()`
