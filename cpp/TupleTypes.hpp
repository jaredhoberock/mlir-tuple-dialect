#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <optional>
#include <TraitOps.hpp>
#include <TraitTypes.hpp>

#define GET_TYPEDEF_CLASSES
#include "TupleTypes.hpp.inc"

namespace mlir::tuple {

// checks that all the element types of tuple_ty have an impl for the given trait
// it not, returns the Type of the first failing element
std::optional<Type> firstElementTypeWithoutImplForTrait(TupleType tuple_ty, mlir::trait::TraitOp traitOp);

inline TupleType getTupleTypeWithFreshPolymorphicElements(MLIRContext* ctx, unsigned int arity) {
  SmallVector<Type> elemPolys;
  for (unsigned int i = 0; i < arity; ++i) {
    elemPolys.push_back(trait::PolyType::fresh(ctx));
  }
  return TupleType::get(ctx, elemPolys);
}

} // end mlir::trait
