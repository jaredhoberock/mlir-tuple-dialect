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

bool AnyTupleType::matches(Type ty, mlir::trait::TraitOp& traitOp) const {
  if (AnyTupleType symbolic_tuple = llvm::dyn_cast<AnyTupleType>(ty)) {
    // ty is a symbolic !tuple.any<[...]>
    // check that it has a trait bound for traitOp
    return symbolic_tuple.hasTraitBound(FlatSymbolRefAttr::get(getContext(), traitOp.getSymName()));
  } else if (TupleType concrete_tuple = llvm::dyn_cast<TupleType>(ty)) {
    // ty is a concrete tuple type
    // check that each of ty's element types has an impl for trait
    return allElementTypesHaveImpl(concrete_tuple, traitOp);
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
