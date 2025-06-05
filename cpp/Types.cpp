#include "Dialect.hpp"
#include "Types.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir-trait-dialect/cpp/Ops.hpp>

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

namespace mlir::tuple {

std::optional<Type> firstElementTypeWithoutImplForTrait(TupleType tuple_ty, mlir::trait::TraitOp traitOp) {
  for (Type element_ty : tuple_ty.getTypes()) {
    if (!traitOp.getImpl(element_ty)) return element_ty;
  }
  return std::nullopt;
}

bool allElementTypesHaveImpl(TupleType tuple_ty, mlir::trait::TraitOp traitOp) {
  return firstElementTypeWithoutImplForTrait(tuple_ty, traitOp) == std::nullopt;
}

SmallVector<trait::TraitOp> AnyTupleType::getTraits(ModuleOp module) const {
  SmallVector<trait::TraitOp> result;
  for (FlatSymbolRefAttr traitBound : getTraitBounds()) {
    auto traitOp = mlir::SymbolTable::lookupNearestSymbolFrom<trait::TraitOp>(module, traitBound);
    if (traitOp)
      result.push_back(traitOp);
  }
  return result;
}

bool AnyTupleType::matches(Type ty, ModuleOp module) const {
  if (AnyTupleType symbolic_tuple = llvm::dyn_cast<AnyTupleType>(ty)) {
    // ty is a symbolic tuple; just check for equality
    // XXX we should maybe check that all of our trait bounds are a subset of those of ty's
    return *this == symbolic_tuple;
  } else if (TupleType concrete_tuple = llvm::dyn_cast<TupleType>(ty)) {
    // ty is a concrete tuple type
    // check that each of ty's element types has an impl for each one of our trait bounds
    for (trait::TraitOp traitOp : getTraits(module)) {
      if (!allElementTypesHaveImpl(concrete_tuple, traitOp))
        return false;
    }
    return true;
  }
  return false;
}

void TupleDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
  >();
}

} // end mlir::tuple
