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

Type PolyType::instantiate(trait::InstantiationMap &inst, uint64_t &idCounter) {
  auto self = cast<trait::GenericTypeInterface>(*this);

  // check memo first - if we've already instantiated this PolyType,
  // return the instance
  if (auto existing = inst.lookup(self))
    return *existing;

  // create and remember a fresh inference var for this poly
  auto fresh = InferenceType::get(getContext(), idCounter++, getInner().getUniqueId());
  inst.bind(self, cast<trait::UnificationTypeInterface>(fresh));
  return fresh;
}

Type PolyType::specializeWith(const trait::SpecializationMap &subst) const {
  // check if this type appears in the substitution
  if (auto replacement = subst.lookup(cast<trait::GenericTypeInterface>(*this)))
    return *replacement;

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
  trait::UnificationMap &subst,
  llvm::function_ref<InFlightDiagnostic()> err) {
  Type self = *this;
  auto selfKey = cast<trait::UnificationTypeInterface>(self);

  // normalize
  other = trait::applySubstitutionToFixedPoint(subst.toTypeMap(), other);

  // first check for trivial equality
  if (self == other) return success();

  // if self is already bound, check consistency
  if (auto existing = subst.lookup(selfKey)) {
    if (*existing != other) {
      if (err) return err() << "inference variable " << self
                            << " already bound to " << *existing
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
    subst.bind(selfKey, other);
    return success();
  }

  // otherwise, reject
  if (err) err() << "type mismatch: expected a tuple type, but found " << other;
  return failure();
}

} // end mlir::tuple
