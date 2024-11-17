# 01 Minimal

## About

LLVMのライブラリとリンクする最低限のサンプルプログラム。

## Build

次のように、必要なLLVMのライブラリをビルド。

```
$ cmake --build . --target MLIRSupport --config Release
```

## Result

```
$ 01-minimal --help
OVERVIEW: 01 minimal

USAGE: 01-minimal.exe [options]

OPTIONS:

Color Options:

  --color     - Use colors in output (default=autodetect)

Generic Options:

  --help      - Display available options (--help-hidden for more)
  --help-list - Display list of available options (--help-list-hidden for more)
  --version   - Display the version of this program
```

## Memo

- `MLIRSupport`: `llvm::cl::ParseCommandLineOptions()`
