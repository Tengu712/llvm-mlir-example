module {
  func.func @main() {
    %A = memref.alloc() : memref<100xf32>
    %B = memref.alloc() : memref<100xf32>
    affine.for %i = 0 to 100 {
        %a = affine.load %A[%i] : memref<100xf32>
        %b = arith.mulf %a, %a : f32
        affine.store %b, %A[%i] : memref<100xf32>
    }
    affine.for %i = 0 to 100 {
        %a = affine.load %A[%i] : memref<100xf32>
        %b = arith.addf %a, %a : f32
        affine.store %b, %B[%i] : memref<100xf32>
    }
    return
  }
}
