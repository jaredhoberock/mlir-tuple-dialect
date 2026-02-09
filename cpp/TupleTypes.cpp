// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Tuple.hpp"
#include "TupleTypes.hpp"
#include <atomic>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <TraitOps.hpp>
#include <TraitTypes.hpp>

#define GET_TYPEDEF_CLASSES
#include "TupleTypes.cpp.inc"

namespace mlir::tuple {


void TupleDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TupleTypes.cpp.inc"
  >();
}


//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

PolyType PolyType::getUnique(MLIRContext* ctx) {
  trait::PolyType inner = trait::PolyType::getUnique(ctx);
  return PolyType::get(ctx, inner);
}

Type PolyType::instantiate(DenseMap<Type,Type> &inst, uint64_t &idCounter) {
  // check memo first - if we've already instantiated this PolyType,
  // return the instance
  if (auto it = inst.find(*this); it != inst.end()) {
    return it->second;
  }

  // create and remember a fresh inference var for this poly
  auto fresh = InferenceType::get(getContext(), idCounter++, getInner().getUniqueId());
  inst[*this] = fresh;
  return fresh;
}

Type PolyType::specializeWith(const DenseMap<Type,Type> &subst) const {
  // check if this type appears in the substitution
  auto it = subst.find(*this);
  if (it != subst.end())
    return it->second;

  // otherwise, specialize the inner type
  trait::PolyType inner = getInner();
  Type specialized = inner.specializeWith(subst);
  if (specialized == inner)
    return *this;

  // if inner is still polymorphic, keep the result wrapped in tuple::PolyType 
  if (auto poly = llvm::dyn_cast<trait::PolyType>(specialized))
    return PolyType::get(getContext(), poly);

  // the inner type specialized to something concrete, return it directly
  return specialized;
}

Type PolyType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  
  if (parser.parseLess())
    return {};

  trait::PolyType inner;
  if (succeeded(parser.parseOptionalKeyword("unique"))) {
    inner = trait::PolyType::getUnique(ctx);
  } else {
    int uniqueId;
    if (parser.parseInteger(uniqueId))
      return {};
    inner = trait::PolyType::get(ctx, uniqueId);
  }

  if (parser.parseGreater())
    return {};

  return PolyType::get(ctx, inner);
}

void PolyType::print(AsmPrinter &printer) const {
  printer << "<" << getInner().getUniqueId() << ">";
}


//===----------------------------------------------------------------------===//
// InferenceType
//===----------------------------------------------------------------------===//

LogicalResult InferenceType::unify(
  Type other,
  ModuleOp /*module*/,
  DenseMap<Type,Type> &subst,
  llvm::function_ref<InFlightDiagnostic()> err) {
  Type self = *this;

  // normalize
  other = trait::applySubstitution(subst, other);

  // first check for trivial equality
  if (self == other) return success();

  // if self is already bound, check consistency
  if (auto it = subst.find(self); it != subst.end()) {
    if (it->second != other) {
      if (err) return err() << "inference variable " << self
                            << " already bound to " << it->second
                            << ", cannot bind to " << other;
      return failure();
    }
    return success();
  }

  // occurs check: forbid T := f(..., T, ...) to avoid cycles
  auto occursIn = [](Type needle, Type haystack) {
    bool hit = false;
    haystack.walk([&](Type t) {
      if (!hit && t == needle) hit = true;
    });
    return hit;
  };

  if (occursIn(self, other)) {
    if (err) err() << "recursive substitution: " << self
                   << " occurs in " << other;
    return failure();
  }

  // accept only tuple-like types
  if (isTupleLike(other)) {
    subst[self] = other;
    return success();
  }

  // otherwise, reject
  if (err) err() << "type mismatch: expected a tuple type, but found " << other;
  return failure();
}

} // end mlir::tuple
