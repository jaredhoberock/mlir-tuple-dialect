// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
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

inline TupleType getTupleTypeWithUniquePolymorphicElements(MLIRContext* ctx, unsigned int arity) {
  SmallVector<Type> elemPolys;
  for (unsigned int i = 0; i < arity; ++i) {
    elemPolys.push_back(trait::PolyType::getUnique(ctx));
  }
  return TupleType::get(ctx, elemPolys);
}

inline bool isTupleLike(Type ty) {
  // check if ty is the symbolic !tuple.poly type
  if (isa<PolyType>(ty)) {
    return true;
  }

  // or the symbolic !tuple.infer type
  if (isa<InferenceType>(ty)) {
    return true;
  }

  // otherwise, check if ty is a concrete tuple type
  if (isa<TupleType>(ty)) {
    return true;
  }

  return false;
}

} // end mlir::trait
