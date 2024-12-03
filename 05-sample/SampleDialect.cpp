#include "SampleDialect.h"

#include "SampleOps.h"

#include "SampleDialect.cpp.inc"

void sample::SampleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "SampleOps.cpp.inc"
  >();
}
