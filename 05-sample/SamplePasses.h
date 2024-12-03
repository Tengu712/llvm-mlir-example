#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace sample {
    std::unique_ptr<mlir::Pass> createLowerToLLVM();
}
