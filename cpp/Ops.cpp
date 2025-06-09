#include "Dialect.hpp"
#include "Ops.hpp"
#include "Types.hpp"
#include <iostream>
#include <mlir/IR/Builders.h>
#include <mlir-trait-dialect/cpp/Types.hpp>

#define GET_OP_CLASSES
#include "Ops.cpp.inc"

namespace mlir::tuple {

LogicalResult AppendOp::verify() {
  MLIRContext* ctx = getContext();

  // tuple.append has two modes
  // 1. concrete mode: input tuple is tuple, result type must also be TupleType
  // 2. polymorphic mode: input tuple is !tuple.any, result type must be !trait.poly
  Type inputTy = getTuple().getType();

  // if input type is tuple<a,b,c>
  if (TupleType concreteTupleTy = dyn_cast<TupleType>(inputTy)) {
    // expected result type is tuple<a,b,c,d>
    SmallVector<Type> resultTypes(concreteTupleTy.getTypes());
    resultTypes.push_back(getElement().getType());
    Type expectedResultTy = TupleType::get(ctx, resultTypes);

    if (expectedResultTy != getResult().getType())
      return emitOpError() << "type mismatch: expected '" << expectedResultTy << "'"
                           << "got '" << getResult().getType() << "'";
    return success();
  }

  // if input type is !tuple.any
  if (AnyTupleType polyTupleTy = dyn_cast<AnyTupleType>(inputTy)) {
    // expected result type is !trait.poly
    if (!isa<trait::PolyType>(getResult().getType()))
      return emitOpError() << "type mismatch: expected '!trait.poly', got '"
                           << getResult().getType() << "'";
    return success();
  }

  return emitError() << "unsupported type for input tuple: '" << inputTy << "'";
}

LogicalResult MakeOp::verify() {
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

YieldOp MapOp::bodyYield() {
  return cast<YieldOp>(getBody().front().back());
}

FunctionType MapOp::getBodyFunctionType() {
  return FunctionType::get(
      getContext(),
      getBody().front().getArgumentTypes(),
      bodyYield().getOperand().getType()
  );
}

FunctionType MapOp::getFunctionTypeForIteration(unsigned int i) {
  SmallVector<Type> argumentTypes;
  for (TupleType input : getInputTupleTypes())
    argumentTypes.push_back(input.getType(i));

  return FunctionType::get(
      getContext(),
      argumentTypes,
      getResultTupleType().getType(i)
  );
}

llvm::DenseMap<Type,Type> MapOp::buildSubstitutionForIteration(unsigned int i) {
  auto bodyTy = getBodyFunctionType();
  auto iterationTy = getFunctionTypeForIteration(i);

  llvm::DenseMap<Type,Type> substitution;
  if (failed(trait::unifyTypes(getLoc(), bodyTy, iterationTy, getOperation()->getParentOfType<ModuleOp>(), substitution))) {
    // this should never happen if MapOp::verifySymbolUses succeeds
    llvm_unreachable("buildSubstitutionForIteration: unification failed");
  }

  return substitution;
}

LogicalResult MapOp::verify() {
  // check that we have at least one input tuple
  if (getInputs().empty())
    return emitOpError("expected at least one input tuple");

  // verify all inputs are TupleType and collect their arities 
  SmallVector<TupleType> inputTupleTypes;
  size_t expectedArity = 0;

  for (auto [i, input] : llvm::enumerate(getInputs())) {
    auto tupleType = dyn_cast<TupleType>(input.getType());
    if (!tupleType)
      return emitOpError("input #") << i << " must be a 'tuple', got "
                                    << input.getType();
    inputTupleTypes.push_back(tupleType);

    // check arity consistency
    if (i == 0) {
      expectedArity = tupleType.size();
    } else if (tupleType.size() != expectedArity) {
      return emitOpError("all input tuples must have the same arity, expected ")
             << expectedArity << " but input #" << i << " has arity "
             << tupleType.size();
    }
  }

  // verify result is a tuple type
  auto resultTupleType = dyn_cast<TupleType>(getResult().getType());
  if (!resultTupleType)
    return emitOpError("result must be a tuple type, got ") << getResult().getType();

  // check result type's arity
  if (resultTupleType.size() != expectedArity)
    return emitOpError("result tuple must have the same arity as input, expected ")
           << expectedArity << " but result tuple has arity "
           << resultTupleType.size();

  // check body block
  Block &bodyBlock = getBody().front();

  // check body argument count matches the number of inputs
  if (bodyBlock.getNumArguments() != getInputs().size())
    return emitOpError("body block must have ") << getInputs().size()
           << " arguments to match the number of tuple inputs, got "
           << bodyBlock.getNumArguments();

  // check that the body block is terminated with YieldOp
  if (bodyBlock.empty())
    return emitOpError("body block cannot be empty");
  if (!isa<YieldOp>(bodyBlock.back()))
    return emitOpError("body block must terminate with `tuple.yield`, got ")
           << bodyBlock.back().getName();

  return success();
}

LogicalResult MapOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // look for the enclosing ModuleOp
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return emitOpError() << "not contained in a module";

  Location loc = getLoc();
  MLIRContext *ctx = getContext();

  // Get input tuple types
  SmallVector<TupleType> inputTupleTypes = getInputTupleTypes();

  // treat the body as if it is a callee and get its function type
  FunctionType calleeTy = getBodyFunctionType();

  // for each tuple element,
  // unify "iteration" i of the body
  for (size_t i = 0; i < getArity(); ++i) {
    // check iteration i like a function call
    // collect a FunctionType for iteration i: these are the call arguments
    FunctionType callerTy = getFunctionTypeForIteration(i);

    // unify each iteration in isolation like a separate function call
    llvm::DenseMap<Type,Type> substitution;
    if (failed(trait::unifyTypes(loc, calleeTy, callerTy, moduleOp, substitution)))
      return failure();
  }

  return success();
}

YieldOp FoldlOp::bodyYield() {
  return cast<YieldOp>(getBody().front().back());
}

llvm::DenseMap<Type,Type> FoldlOp::buildSubstitutionForIteration(
    unsigned int i, 
    Type resultTypeOfPreviousIteration) {
  Location loc = getLoc();
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();

  Type expectedTy0 = getBody().front().getArgumentTypes()[0];
  Type expectedTy1 = getBody().front().getArgumentTypes()[1];

  llvm::DenseMap<Type,Type> substitution;

  // we unify the resultTypeOfPreviousIteration with the body's zeroth block parameter type
  if (failed(trait::unifyTypes(loc, expectedTy0, resultTypeOfPreviousIteration, module, substitution))) {
    // this should never happen if FoldlOp::verifySymbolUses succeeds
    llvm_unreachable("buildSubstitutionForIteration: unification failed");
  }

  // we unify the ith tuple element type with the body's first block parameter type
  if (failed(trait::unifyTypes(loc, expectedTy1, getTupleType().getType(i), module, substitution))) {
    // this should never happen if FoldlOp::verifySymbolUses succeeds
    llvm_unreachable("buildSubstitutionForIteration: unification failed");
  }

  return substitution;
}

LogicalResult FoldlOp::verify() {
  // check body block
  Block &bodyBlock = getBody().front();

  // check body argument count is 2
  if (bodyBlock.getNumArguments() != 2)
    return emitOpError("body block must have 2 arguments, got ")
           << bodyBlock.getNumArguments();

  // check that the body block is terminated with YieldOp
  if (bodyBlock.empty())
    return emitOpError("body block cannot be empty");
  if (!isa<YieldOp>(bodyBlock.back()))
    return emitOpError("body block must terminate with `tuple.yield`, got ")
           << bodyBlock.back().getName();

  return success();
}

LogicalResult FoldlOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // look for the enclosing ModuleOp
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  if (!moduleOp)
    return emitOpError("not contained in a module");

  Location loc = getLoc();
  MLIRContext* ctx = getContext();

  // get input tuple type
  TupleType tupleTy = getTupleType();

  // for each tuple element,
  // unify iteration i of the body *in isolation*
  // in other words, we use a separate substitution for each iteration
  Type previousIterationResultType = getInit().getType();
  for (size_t i = 0; i < getArity(); ++i) {
    // unify types involved in iteration i
    Type foundTy0 = previousIterationResultType;
    Type foundTy1 = tupleTy.getType(i);

    Type expectedTy0 = getBody().front().getArgumentTypes()[0];
    Type expectedTy1 = getBody().front().getArgumentTypes()[1];

    llvm::DenseMap<Type,Type> substitution;
    if (failed(trait::unifyTypes(loc, expectedTy0, foundTy0, moduleOp, substitution)))
      return failure();
    if (failed(trait::unifyTypes(loc, expectedTy1, foundTy1, moduleOp, substitution)))
      return failure();

    // update the previous result type by applying the substitution to the body's yield type
    previousIterationResultType = trait::applySubstitution(substitution, bodyYield().getOperand().getType());
  }

  // unify the final result type with the ascribed result type
  llvm::DenseMap<Type,Type> substitution;
  if (failed(trait::unifyTypes(loc, previousIterationResultType, getResult().getType(),
                               moduleOp, substitution)))
    return failure();

  return success();
}

}
