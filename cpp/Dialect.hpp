#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "Dialect.hpp.inc"
#include "Types.hpp"

namespace mlir::tuple {

inline bool isTuple(Type ty) {
  // check if ty is the symbolic !tuple.any_tuple type
  if (isa<AnyTupleType>(ty)) {
    return true;
  }

  // otherwise, check if ty is a concrete tuple type
  if (isa<TupleType>(ty)) {
    return true;
  }

  return false;
}

}
