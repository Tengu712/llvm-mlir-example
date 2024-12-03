#ifndef SAMPLE
#define SAMPLE

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Sample_Dialect : Dialect {
  let name = "sample";

  let summary = "This is sample dialect.";

  let description = [{
    This is description.
  }];

  let cppNamespace = "sample";
}

class Sample_Op<string mnemonic, list<Trait> traits = []> :
    Op<Sample_Dialect, mnemonic, traits>;

def ConstantOp : Sample_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    ```mlir
      %0 = "sample.constant"()
      { value = 13 : i32 }
      : () -> i32
    ```
  }];

  let arguments = (ins I32Attr:$value);
  let results = (outs I32);

  let assemblyFormat = "$value attr-dict `:` type(results)";
}

#endif // SAMPLE
