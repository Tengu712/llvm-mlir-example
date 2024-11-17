import glob
import os
import subprocess
import sys

SRC_DIR_NAMES = ["01-minimal", "02-noop", "03-to-llvmir"]
INC_PATH_ENVS = [
    "LLVM_LLVM_INCLUDE_PATH",
    "LLVM_MLIR_INCLUDE_PATH",
    "LLVM_LLVM_BUILD_INCLUDE_PATH",
    "LLVM_MLIR_BUILD_INCLUDE_PATH"
]
LIB_PATH_ENV = "LLVM_LIBRARY_PATH"
OPTIONS = [
    "/EHsc",
    "/std:c++17",
    "/GL",
    "/Gy",
    "/O2",
    "/Zc:inline",
    "/DNDEBUG",
    "/D_CONSOLE",
    "/D_UNICODE",
    "/DUNICODE",
    "/source-charset:utf-8",
    "/Gd",
    "/Oi",
    "/MD",
    "/FC",
    "/nologo",
    "/diagnostics:column",
    "/D_srcDirNameLENCE_NONFLOATING_COMPLEX_DEPRECATION_WARNING"
]

# コマンドライン引数からビルドするサンプルプログラムのディレクトリ名を取得
args = sys.argv
if len(args) < 2:
    print("usage: py build-with-msvc.py <target-number>")
    exit(1)
srcDirNameIdx = int(args[1]) - 1
if srcDirNameIdx < 0 or srcDirNameIdx >= len(SRC_DIR_NAMES):
    print("fatal: target-number must be an integer between 1 and " + str(len(SRC_DIR_NAMES)) + " but " + args[1] + " is passed.")
    exit(1)
srcDirName = SRC_DIR_NAMES[srcDirNameIdx]

# 各種パスを取得
rtd = os.path.dirname(__file__).replace("\\", "/")
srd = rtd + "/" + srcDirName + "/"
tgd = rtd + "/build/"
lbd = os.environ.get(LIB_PATH_ENV)

# buildディレクトリへカレントディレクトリを移動
if not os.path.exists(tgd):
    os.mkdir(tgd)
os.chdir(tgd)

# ビルド
incs = ["-I" + os.environ.get(n) for n in INC_PATH_ENVS]
cpps = glob.glob(srd + "**/*.cpp", recursive=True)
libs = [n for n in os.listdir(lbd) if n.endswith(".lib")]
if subprocess.run(["cl", "/Fe:" + srcDirName, *OPTIONS, *incs, *cpps, "/link", "ntdll.lib", "/LIBPATH:" + lbd, *libs]).returncode != 0:
    exit(1)

# .objファイルをすべて削除
objs = glob.glob("*.obj")
for obj in objs:
    try:
        os.remove(obj)
    except Exception as e: pass
