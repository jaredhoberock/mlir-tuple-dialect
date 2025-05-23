#include "c_api.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::tuple;

extern "C" {

void tupleRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<TupleDialect>();
}

MlirOperation tupleConstantOpCreate(MlirLocation loc, MlirValue* elements, intptr_t nElements) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value, 4> elementVals;
  for (intptr_t i = 0; i < nElements; ++i)
    elementVals.push_back(unwrap(elements[i]));

  auto op = builder.create<ConstantOp>(
      unwrap(loc),
      elementVals
  );
  return wrap(op.getOperation());
}

MlirOperation tupleGetOpCreate(MlirLocation loc, MlirValue tuple, int64_t index) {
  MLIRContext* ctx = unwrap(loc)->getContext();

  OpBuilder builder(ctx);
  auto op = builder.create<GetOp>(
    unwrap(loc),
    unwrap(tuple),
    index
  );

  return wrap(op.getOperation());
}

MlirOperation tupleCmpOpCreate(MlirLocation loc,
                               TupleCmpPredicate predicate,
                               MlirValue lhs,
                               MlirValue rhs) {
  MLIRContext *ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  auto cppPredicate = static_cast<tuple::CmpPredicate>(predicate);
  auto op = builder.create<CmpOp>(
    unwrap(loc),
    cppPredicate,
    unwrap(lhs),
    unwrap(rhs)
  );
  return wrap(op.getOperation());
}

} // end extern "C"
