#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <mlir/IR/Builders.h>
#include <iostream>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace mlir::tuple {

LogicalResult ConstantOp::verify() {
  auto tupleTy = dyn_cast<TupleType>(getResult().getType());
  if (!tupleTy)
    return emitOpError("result must be a tuple type");
  if (tupleTy.size() != getNumOperands())
    return emitOpError("operand/result arity mismatch");
  return success();
}

LogicalResult GetOp::verify() {
  // get the TupleType of the operand
  TupleType tupleTy = getTupleType();
  int64_t index = getIndex().getSExtValue();

  // bounds check
  auto elementTypes = tupleTy.getTypes();
  if (index < 0 || index >= elementTypes.size()) {
    return emitOpError() << "index " << index
                         << " out of bounds for tuple of size "
                         << elementTypes.size();
  }

  // check that the result type matches the element type
  Type expectedTy = elementTypes[index];
  if (getResult().getType() != expectedTy) {
    return emitOpError() << "result type " << getResult().getType()
                         << " does not match tuple element type " << expectedTy
                         << " at index " << index;
  }

  return success();
}

StringRef CmpOp::getTraitName() {
  if (getPredicate() == CmpPredicate::eq ||
      getPredicate() == CmpPredicate::ne) {
    return "PartialEq";
  }
  return "PartialOrd";
}

FlatSymbolRefAttr CmpOp::getTraitRefAttr() {
  return FlatSymbolRefAttr::get(getContext(), getTraitName());
}

StringRef CmpOp::getMethodName() {
  switch (getPredicate()) {
    case CmpPredicate::eq:
      return "eq";
    case CmpPredicate::ne:
      return "ne";
    case CmpPredicate::lt:
      return "lt";
    case CmpPredicate::le:
      return "le";
    case CmpPredicate::gt:
      return "gt";
    case CmpPredicate::ge:
      return "ge";
  }
  return {};
}

mlir::trait::TraitOp CmpOp::getTrait() {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    emitOpError() << "not inside of a module";
    return nullptr;
  }
  return mlir::SymbolTable::lookupNearestSymbolFrom<mlir::trait::TraitOp>(module, getTraitRefAttr());
}

LogicalResult CmpOp::verifyTupleTypeHasImplFor(mlir::trait::TraitOp traitOp, Type tuple_ty) {
  LogicalResult result = success();

  if (auto concrete_tuple_ty = dyn_cast<TupleType>(tuple_ty)) {
    // tuple_ty is a concrete tuple<...> type
    // recursively check for impls for each element type
    for (Type element_ty : concrete_tuple_ty.getTypes()) {
      if (isa<TupleType>(element_ty) or isa<AnyTupleType>(element_ty)) {
        if (failed(verifyTupleTypeHasImplFor(traitOp, element_ty))) {
          result = failure();
        }
      } else {
        // element_ty is not a tuple; look for an impl
        if (!traitOp.getImpl(element_ty)) {
          result = emitOpError() << "tuple element type " << element_ty << " does not have a trait.impl for trait '" << getTraitRefAttr() << "'";
        }
      }
    }
  } else if (auto symbolic_tuple_ty = dyn_cast<AnyTupleType>(tuple_ty)) {
    // typle_ty is a symbolic !tuple.any type
    // check for a trait bound
    if (!symbolic_tuple_ty.hasTraitBound(getTraitRefAttr())) {
      result = emitOpError() << symbolic_tuple_ty << " does not have a trait bound for trait '" << getTraitRefAttr() << "'";
    }
  } else {
    llvm_unreachable("CmpOp::verifyTupleTypeHasImplFor: expected tuple type to be either 'tuple' or '!tuple.any'");
  }

  return result;
}

LogicalResult CmpOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // if the TupleType is concrete and empty, the trait needn't exist at all
  if (auto tupleTy = dyn_cast<TupleType>(getLhs().getType())) {
    if (tupleTy.getTypes().empty()) {
      return success();
    }
  }

  // look up the trait.trait we need
  auto traitOp = getTrait();
  if (!traitOp) {
    return emitOpError() << "couldn't find trait.trait '" << getTraitRefAttr() << "'";
  }

  // make sure the trait has the method we'll need as well
  if (!traitOp.hasMethod(getMethodName())) {
    return emitOpError() << "couldn't find method '@" << getMethodName() << "' in trait.trait '" << getTraitRefAttr() << "'";
  }

  // verify that the operand type has impls of the trait we will use during lowering
  return verifyTupleTypeHasImplFor(traitOp, getLhs().getType());
}

}
