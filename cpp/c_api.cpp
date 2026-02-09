// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "c_api.h"
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace mlir::tuple;

extern "C" {

void tupleRegisterDialect(MlirContext context) {
  unwrap(context)->loadDialect<TupleDialect>();
}

MlirOperation tupleMakeOpCreate(MlirLocation loc, MlirValue* elements, intptr_t nElements) {
  MLIRContext* ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  SmallVector<Value, 4> elementVals;
  for (intptr_t i = 0; i < nElements; ++i)
    elementVals.push_back(unwrap(elements[i]));

  auto op = builder.create<MakeOp>(
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
                               MlirValue rhs,
                               MlirValue claims) {
  MLIRContext *ctx = unwrap(loc)->getContext();
  OpBuilder builder(ctx);

  auto cppPredicate = static_cast<tuple::CmpPredicate>(predicate);

  auto op = mlirValueIsNull(claims)
    ? builder.create<CmpOp>(unwrap(loc), cppPredicate, unwrap(lhs), unwrap(rhs))
    : builder.create<CmpOp>(unwrap(loc), cppPredicate, unwrap(lhs), unwrap(rhs), unwrap(claims));

  return wrap(op.getOperation());
}

} // end extern "C"
