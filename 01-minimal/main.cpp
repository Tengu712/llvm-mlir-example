#include <llvm/Support/CommandLine.h>

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "01 minimal\n");
  return 0;
}
