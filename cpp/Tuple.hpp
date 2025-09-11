#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "Tuple.hpp.inc"
#include "TupleTypes.hpp"

namespace mlir::tuple {

inline bool isTupleLike(Type ty) {
  // check if ty is the symbolic !tuple.poly type
  if (isa<PolyType>(ty)) {
    return true;
  }

  // otherwise, check if ty is a concrete tuple type
  if (isa<TupleType>(ty)) {
    return true;
  }

  return false;
}

}
