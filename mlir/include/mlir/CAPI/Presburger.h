#ifndef B8136BCB_5471_4828_990A_D00FC2B644CD
#define B8136BCB_5471_4828_990A_D00FC2B644CD

#include "mlir-c/Presburger.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/CAPI/Wrap.h"

static inline MlirPresburgerIntegerRelation
wrap(mlir::presburger::IntegerRelation *cpp) {
  return MlirPresburgerIntegerRelation{cpp->getAsOpaquePointer()};
}
static inline mlir::presburger::IntegerRelation *
unwrap(MlirPresburgerIntegerRelation c) {
  return mlir::presburger::IntegerRelation::getFromOpaquePointer(c.ptr);
}

#endif /* B8136BCB_5471_4828_990A_D00FC2B644CD */
