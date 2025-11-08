#include "Tuple.hpp"
#include "TupleOps.hpp"
#include "TupleTypes.hpp"
#include <iostream>
#include <mlir/IR/Builders.h>
#include <TraitTypes.hpp>

#define GET_OP_CLASSES
#include "TupleOps.cpp.inc"

namespace mlir::tuple {

LogicalResult AllOp::verify() {
  // body must exist, have exactly 1 arg, and end with tuple.yield
  Block &body = getBody().front();
  unsigned numExpectedArgs = 1;
  if (body.getNumArguments() != numExpectedArgs)
    return emitOpError() << "body block must have exactly one argument, got "
                         << body.getNumArguments();

  if (body.empty())
    return emitOpError("body block cannot be empty");

  if (!isa<YieldOp>(body.back()))
    return emitOpError("body block must terminate with `tuple.yield`, got ")
           << body.back().getName();

  // ensure region yields i1
  Type yieldedTy = bodyYield().getResult().getType();
  if (!yieldedTy.isInteger(1))
    return emitOpError() << "body block must yield i1, got " << yieldedTy;
  return success();
}

YieldOp AllOp::bodyYield() {
  return cast<YieldOp>(getBody().front().back());
}

FunctionType AllOp::getBodyFunctionType() {
  return FunctionType::get(
      getContext(),
      getBody().front().getArgumentTypes(),
      bodyYield().getOperand().getType()
  );
}

FunctionType AllOp::getFunctionTypeForIteration(unsigned int i) {
  auto inputTupleType = getInputTupleTypeWithKnownArity();
  if (failed(inputTupleType))
    llvm_unreachable("AllOp::getFunctionTypeForIteration: input must be TupleType");

  return FunctionType::get(
      getContext(),
      {inputTupleType->getType(i)},
      bodyYield().getOperand().getType()
  );
}

LogicalResult AllOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // must be inside a module
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return emitOpError("not contained in a module");

  if (auto N = getArity())
    return verifySymbolUsesWithKnownArity(moduleOp, *N);
  return verifySymbolUsesWithUnknownArity(moduleOp);
}

LogicalResult AllOp::verifySymbolUsesWithKnownArity(ModuleOp module, unsigned arity) {
  auto err = [&]{ return emitOpError(); };

  // treat the body as a function: (Eformal) -> i1
  FunctionType calleeTy = getBodyFunctionType();

  // for each tuple element i, the "caller" is: (Eactual) -> i1
  for (unsigned i = 0; i < arity; ++i) {
    FunctionType callerTy = getFunctionTypeForIteration(i);

    // unify body formal with actual for this iteration
    if (failed(trait::buildSpecializationSubstitution(calleeTy, callerTy, module, err)))
      return failure();
  }
  return success();
}

LogicalResult AllOp::verifySymbolUsesWithUnknownArity(ModuleOp module) {
  // Callee type is (Eformal) -> i1. Ensure Eformal is purely polymorphic.
  FunctionType calleeTy = getBodyFunctionType();
  Type Eformal = calleeTy.getInput(0);
  if (!trait::isPurelyPolymorphicType(Eformal))
    return emitOpError()
           << "body argument must be purely polymorphic (all leaves e.g. '!trait.poly'); got "
           << Eformal;

  // Yield/result is i1 and already checked in verify().
  return success();
}

LogicalResult AppendOp::verify() {
  MLIRContext* ctx = getContext();

  // tuple.append has two modes
  // 1. concrete mode: input tuple is tuple, result type must also be TupleType
  // 2. polymorphic mode: input tuple is !tuple.poly, result type must be !trait.poly
  //    XXX TODO seems like the result type could also be !tuple.poly
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

  // if input type is !tuple.poly
  if (PolyType polyTupleTy = dyn_cast<PolyType>(inputTy)) {
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


//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

FailureOr<std::optional<unsigned>> CmpOp::verifyArity(llvm::function_ref<InFlightDiagnostic()> err) {
  std::optional<unsigned> seen;

  auto check = [&](Type t, StringRef which) {
    if (!isTupleLike(t)) {
      if (err) err() << which << " must be a tuple type, got " << t;
      return failure();
    }
    if (auto tt = dyn_cast<TupleType>(t)) {
      unsigned n = tt.size();
      if (seen && *seen != n) {
        if (err) err() << "arity mismatch: " << which << " has arity " << n
                       << " but another operand has arity " << *seen;
        return failure();
      }
      seen = n;
    }
    return success();
  };

  if (failed(check(getLhs().getType(), "lhs"))) return failure();
  if (failed(check(getRhs().getType(), "rhs"))) return failure();
  if (Value c = getClaims())
    if (failed(check(c.getType(), "claims"))) return failure();

  return seen;
}

LogicalResult CmpOp::verify() {
  auto errFn = [&]{ return emitOpError(); };

  // verify arity
  if (failed(verifyArity(errFn)))
    return failure();

  Type L = getLhs().getType();
  Type R = getRhs().getType();
  Value claims = getClaims();

  // classify monomorphic vs polymorphic mode
  bool monomorphicMode = trait::isMonomorphicType(L) && trait::isMonomorphicType(R);

  if (monomorphicMode) {
    // in monomorphic mode, L & R must be identical
    if (L != R)
      return emitOpError() << "type mismatch: lhs and rhs must have the same type; expected "
                           << L << ", but got " << R;

    // claims is optional, but if present, must be monomorphic
    if (claims) {
      Type C = claims.getType();
      if (!trait::isMonomorphicType(C))
        return emitOpError() << "type mismatch: claims must be monomorphic when tuple operands are monomorphic; got "
                             << C;
    }

    return success();
  }

  // in polymorphic mode, claims is required
  if (!claims)
    return emitOpError() << "claims operand is required when either tuple input is polymorphic";

  return success();
}

static FailureOr<Type> getFormalClaimsTypeForCmpOp(
  MLIRContext* ctx,
  Type L,
  Type R,
  FlatSymbolRefAttr traitRef,
  std::optional<unsigned> arity,
  llvm::function_ref<InFlightDiagnostic()> err) {

  // require tuple-like operands
  if (!isTupleLike(L) || !isTupleLike(R)) {
    if (err) err() << "lhs and rhs must be tuple types; got " << L << " and " << R;
    return failure();
  }

  // unknown arity -> expect !tuple.poly
  if (!arity)
    return tuple::PolyType::getUnique(ctx);

  auto LT = dyn_cast<TupleType>(L);
  auto RT = dyn_cast<TupleType>(R);

  // if exactly one side is concrete, synthesize a tuple with unique poly elements for the other
  if (!LT)
    LT = getTupleTypeWithUniquePolymorphicElements(ctx, *arity);
  else if (!RT)
    RT = getTupleTypeWithUniquePolymorphicElements(ctx, *arity);

  // now both must be TupleType
  assert(LT && RT && "Expected both LT and RT to be TupleType");

  if (LT.size() != *arity || RT.size() != *arity) {
    if (err) err() << "arity mismatch: lhs has " << LT.size()
                   << ", rhs has " << RT.size()
                   << ", but expected arity is " << *arity;
    return failure();
  }

  // map traitRef over the elements of LT & RT
  SmallVector<Type> elems;
  elems.reserve(*arity);
  for (auto [Li, Ri] : llvm::zip(LT.getTypes(), RT.getTypes())) {
    Type Ci = trait::ClaimType::get(ctx, traitRef, {Li, Ri});
    elems.push_back(Ci);
  }

  return TupleType::get(ctx, elems);
}

LogicalResult CmpOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // if LHS is a concrete empty tuple, allow trivially
  if (auto tupleTy = dyn_cast<TupleType>(getLhs().getType())) {
    if (tupleTy.getTypes().empty()) {
      return success();
    }
  }

  auto errFn = [&]{ return emitOpError(); };

  // look up the base trait we need
  auto trait = getTrait(errFn);
  if (failed(trait))
    return failure();

  // make sure the trait has the method we'll need as well
  if (failed(trait->getMethod(getMethodName(), errFn)))
    return failure();

  // if no claims were provided, this is the monomorphic path
  Value claims = getClaims();
  if (!claims) {
    // nothing more to check
    return success();
  }

  // check the type of claims
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) return emitOpError() << "not in a module";

  auto formalClaimsTy = getFormalClaimsTypeForCmpOp(
    getContext(),
    getLhs().getType(),
    getRhs().getType(),
    getTraitRefAttr(),
    getArity(),
    errFn
  );
  if (failed(formalClaimsTy)) return failure();

  // check that the types can unify
  if (failed(trait::buildSpecializationSubstitution(*formalClaimsTy, claims.getType(), module, errFn)))
    return failure();

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

ParseResult CmpOp::parse(OpAsmParser &p, OperationState &st) {
  // parse <predicate>
  auto loc = p.getCurrentLocation();
  StringRef predTok;
  if (p.parseKeyword(&predTok))
    return p.emitError(loc, "expected tuple.cmp predicate keyword");

  loc = p.getCurrentLocation();
  auto maybePred = symbolizeCmpPredicate(predTok);
  if (!maybePred)
    return p.emitError(loc) << "unknown tuple.cmp predicate '" << predTok << "'";

  auto predAttr = CmpPredicateAttr::get(p.getContext(), *maybePred);
  st.addAttribute("predicate", predAttr);

  // parse ',' %lhs ',' %rhs (',' %claims)?
  OpAsmParser::UnresolvedOperand lhs, rhs, claims;
  bool hasClaimsOperand = false;

  if (p.parseComma() || p.parseOperand(lhs) || p.parseComma() || p.parseOperand(rhs))
    return failure();

  if (succeeded(p.parseOptionalComma())) {
    hasClaimsOperand = true;
    auto loc = p.getCurrentLocation();
    if (p.parseOperand(claims))
      return p.emitError(loc, "expected claims operand after ','");
  }

  // parse attrs
  if (p.parseOptionalAttrDict(st.attributes))
    return failure();

  // parse ':' !L ',' !R (',' !C)?
  if (p.parseColon())
    return failure();

  Type lhsTy, rhsTy, claimsTy;
  if (p.parseType(lhsTy) || p.parseComma() || p.parseType(rhsTy))
    return failure();

  bool hasClaimsType = false;
  SMLoc claimsTyLoc;
  if (succeeded(p.parseOptionalComma())) {
    hasClaimsType = true;
    claimsTyLoc = p.getCurrentLocation();
    if (p.parseType(claimsTy)) return failure();
  }

  // coupling rules: either both claims op & type present or neither
  if (hasClaimsOperand != hasClaimsType) {
    if (hasClaimsType) {
      return p.emitError(claimsTyLoc,
        "claims type provided without claims operand");
    } else {
      auto loc = claims.location.isValid() ? claims.location : p.getCurrentLocation();
      return p.emitError(loc,
        "claims operand provided without claims type");
    }
  }

  // result type: i1
  st.addTypes(p.getBuilder().getI1Type());

  // resolve operands
  SmallVector<OpAsmParser::UnresolvedOperand,3> ops = {lhs, rhs};
  SmallVector<Type, 3> types = {lhsTy, rhsTy};
  if (hasClaimsOperand) {
    ops.push_back(claims);
    types.push_back(claimsTy);
  }

  loc = lhs.location.isValid() ? lhs.location : p.getCurrentLocation();
  if (p.resolveOperands(ops, types, loc, st.operands))
    return failure();

  return success();
}

void CmpOp::print(OpAsmPrinter &p) {
  // <pred>, %lhs, %rhs[, %claims] attrs : !L, !R[, !C]
  p << " " << getPredicate() << ", " << getLhs() << ", " << getRhs();
  if (auto c = getClaims()) p << ", " << c;
  p.printOptionalAttrDict((*this)->getAttrs(), /*elided=*/{"predicate"});
  p << " : " << getLhs().getType() << ", " << getRhs().getType();
  if (auto c = getClaims()) p << ", " << c.getType();
}

static FailureOr<mlir::trait::TraitOp> getTraitInModule(
  ModuleOp module,
  FlatSymbolRefAttr traitRef,
  llvm::function_ref<InFlightDiagnostic()> err) {
  auto traitOp = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::trait::TraitOp>(module, traitRef);
  if (!traitOp) {
    if (err) err() << "couldn't find trait.trait '" << traitRef << "'";
    return failure();
  }
  return traitOp;
}

FailureOr<mlir::trait::TraitOp> CmpOp::getTrait(llvm::function_ref<InFlightDiagnostic()> err) {
  ModuleOp module = getOperation()->getParentOfType<ModuleOp>();
  if (!module) {
    if (err) err() << "not inside of a module";
    return failure();
  }
  return getTraitInModule(module, getTraitRefAttr(), err);
}


//===----------------------------------------------------------------------===//
// FoldlOp
//===----------------------------------------------------------------------===//

FailureOr<std::optional<unsigned>> FoldlOp::verifyArity(llvm::function_ref<InFlightDiagnostic()> err) {
  std::optional<unsigned> seen;

  for (auto [idx, v] : llvm::enumerate(getInputs())) {
    Type ty = v.getType();

    // inputs must be some kind of tuple. if not, error
    if (!isTupleLike(ty)) {
      if (err) err() << "input #" << idx << " must be a tuple type, got " << ty;
      return failure();
    }

    // concrete tuples must agree on arity
    if (auto tup = dyn_cast<TupleType>(ty)) {
      unsigned n = tup.size();
      if (seen && *seen != n) {
        if (err) err() << "arity mismatch: input #" << idx
                       << " has arity " << n << " but a previous input has arity "
                       << *seen;
        return failure();
      }
      seen = n;
    }
  }

  return seen;
}

LogicalResult FoldlOp::verify() {
  // check that we have at least one input tuple
  if (getInputs().empty())
    return emitOpError("expected at least one input tuple");

  // body must exist, have exactly (1 + #inputs) args, and end with tuple.yield
  Block &body = getBody().front();
  unsigned numExpectedArgs = 1 + getInputs().size();
  if (body.getNumArguments() != numExpectedArgs)
    return emitOpError() << "body block must have "
                         << numExpectedArgs << " arguments (accumulator + one per input tuple), got "
                         << body.getNumArguments();

  if (body.empty())
    return emitOpError("body block cannot be empty");

  if (!isa<YieldOp>(body.back()))
    return emitOpError("body block must terminate with `tuple.yield`, got ")
           << body.back().getName();

  // finally verify that all concrete tuples agree on arity
  return verifyArity([&] {
    return emitOpError();
  });
}

LogicalResult FoldlOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // must be inside a module
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return emitOpError("not contained in a module");

  if (auto N = getArity())
    return verifySymbolUsesWithKnownArity(moduleOp, *N);
  return verifySymbolUsesWithUnknownArity(moduleOp);
}

LogicalResult FoldlOp::verifySymbolUsesWithKnownArity(ModuleOp module, unsigned arity) {
  auto err = [&]{ return emitOpError(); };

  // treat the body like a function with type (A, E1..Em) -> R
  FunctionType calleeTy = getBodyFunctionType();
  Type R = calleeTy.getResult(0);

  // thread the accumulator type across iterations
  Type prev = getInit().getType();
  for (unsigned i = 0; i < arity; ++i) {
    FunctionType callerTy = getFunctionTypeForIteration(i, prev);

    // unify each iteration in isolation as if it was a separate function call
    auto subst = trait::buildSpecializationSubstitution(calleeTy, callerTy, module, err);
    if (failed(subst)) return failure();

    // update the previous result type by applying the substitution
    // to the body's result type
    prev = trait::applySubstitutionToFixedPoint(*subst, R);
  }

  // unify the formal result type with the actual final result type
  return trait::buildSpecializationSubstitution(getResult().getType(), prev, module, err);
}

LogicalResult FoldlOp::verifySymbolUsesWithUnknownArity(ModuleOp module) {
  auto err = [&] { return emitOpError(); };

  // treat the body like a function with type:
  // (accFormal, E1..Em) -> yieldFormal
  FunctionType calleeTy = getBodyFunctionType();
  Type accFormal        = calleeTy.getInput(0);  // formal type of %acc
  Type yieldFormal      = calleeTy.getResult(0); // formal yield type
  Type initActual       = getInit().getType();   // actual type of %init
  Type resultFormal     = getResult().getType(); // op's formal result type

  // must be able to specialize the formal acc with the actual init type
  if (failed(trait::buildSpecializationSubstitution(accFormal, initActual, module, err)))
    return failure();

  // every non-accumulator body arg type must be purely polymorphic
  for (auto [i, Ei] : llvm::enumerate(calleeTy.getInputs().drop_front())) {
    if (!trait::isPurelyPolymorphicType(Ei))
      return err() << "non-accumulator body argument #" << (i + 1)
                   << " must be purely polymorphic (all leaves are e.g. '!trait.poly'); got "
                   << Ei;
  }

  // closure: one step of the body must preserve the accumulator shape
  if (failed(trait::buildSpecializationSubstitution(accFormal, yieldFormal, module, err)))
    return failure();

  // op result consistency: op's formal result must match the accumulator
  return trait::buildSpecializationSubstitution(resultFormal, accFormal, module, err);
}

YieldOp FoldlOp::bodyYield() {
  return cast<YieldOp>(getBody().front().back());
}

FunctionType FoldlOp::getBodyFunctionType() {
  return FunctionType::get(
      getContext(),
      getBody().front().getArgumentTypes(),
      bodyYield().getOperand().getType()
  );
}

FunctionType FoldlOp::getFunctionTypeForIteration(
    unsigned int i,
    Type resultTypeOfPreviousIteration) {
  auto inputTupleTypes = getInputTypesAsTupleTypes();
  if (failed(inputTupleTypes))
    llvm_unreachable("FoldlOp::getFunctionTypeForIteration: inputs must be TupleTypes");

  SmallVector<Type> argumentTypes;
  argumentTypes.push_back(resultTypeOfPreviousIteration);
  for (TupleType input : *inputTupleTypes)
    argumentTypes.push_back(input.getType(i));

  return FunctionType::get(
      getContext(),
      argumentTypes,
      bodyYield().getOperand().getType()
  );
}

llvm::DenseMap<Type,Type> FoldlOp::buildSubstitutionForIteration(
    unsigned int i, 
    Type resultTypeOfPreviousIteration) {
  assert(inputTypesAreTupleTypes() && "FoldlOp::buildSubstitutionForIteration: inputs must be TupleType");

  auto bodyTy = getBodyFunctionType();
  auto iterationTy = getFunctionTypeForIteration(i, resultTypeOfPreviousIteration);

  auto module = getOperation()->getParentOfType<ModuleOp>();
  auto subst = trait::buildSpecializationSubstitution(bodyTy, iterationTy, module);
  if (failed(subst)) {
    // this should never happen if FoldlOp::verifySymbolUses succeeds
    llvm_unreachable("buildSubstitutionForIteration: unification failed");
  }
  return *subst;
}


//===----------------------------------------------------------------------===//
// MapOp
//===----------------------------------------------------------------------===//

FailureOr<std::optional<unsigned>> MapOp::verifyArity(llvm::function_ref<InFlightDiagnostic()> err) {
  std::optional<unsigned> seen;

  for (auto [idx, v] : llvm::enumerate(getInputs())) {
    Type ty = v.getType();

    // inputs must be some kind of tuple. if not, error
    if (!isTupleLike(ty)) {
      if (err) err() << "input #" << idx << " must be a tuple type, got " << ty;
      return failure();
    }

    // concrete tuples must agree on arity
    if (auto tup = dyn_cast<TupleType>(ty)) {
      unsigned n = tup.size();
      if (seen && *seen != n) {
        if (err) err() << "arity mismatch: input #" << idx
                       << " has arity " << n << " but a previous input has arity "
                       << *seen;
        return failure();
      }
      seen = n;
    }
  }

  return seen;
}

LogicalResult MapOp::verify() {
  // check that we have at least one input tuple
  if (getInputs().empty())
    return emitOpError("expected at least one input tuple");

  // body must exist, have exactly #inputs args, and end with tuple.yield
  Block &body = getBody().front();
  unsigned numExpectedArgs = getInputs().size();
  if (body.getNumArguments() != numExpectedArgs)
    return emitOpError() << "body block must have "
                         << numExpectedArgs << " arguments (one per input tuple), got "
                         << body.getNumArguments();

  if (body.empty())
    return emitOpError("body block cannot be empty");

  if (!isa<YieldOp>(body.back()))
    return emitOpError("body block must terminate with `tuple.yield`, got ")
           << body.back().getName();

  // verify that all concrete tuples agree on arity
  FailureOr<std::optional<unsigned>> maybeArity = verifyArity([&] {
    return emitOpError();
  });

  if (failed(maybeArity))
    return failure();

  if (*maybeArity) {
    // known arity path: result must be a concrete tuple with that arity
    auto resTup = dyn_cast<TupleType>(getResult().getType());
    if (!resTup)
      return emitOpError("result must be a tuple type, got ")
             << getResult().getType();
    if (resTup.size() != **maybeArity)
      return emitOpError("arity mismatch: result tuple has arity ") << resTup.size()
             << ", but input tuples have arity " << **maybeArity;
  } else {
    // unknown arity path: result must be !tuple.poly
    if (!isa<tuple::PolyType>(getResult().getType()))
      return emitOpError("result must be !tuple.poly when arity is unknown");
  }

  return success();
}

LogicalResult MapOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // must be inside a module
  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("not contained in a module");

  if (auto N = getArity())
    return verifySymbolUsesWithKnownArity(module, *N);
  return verifySymbolUsesWithUnknownArity();
}

LogicalResult MapOp::verifySymbolUsesWithKnownArity(ModuleOp module,
                                                    unsigned arity) {
  auto err = [&]{ return emitOpError(); };

  // treat the body like a function with type (E1..Em) -> R
  FunctionType calleeTy = getBodyFunctionType();

  // check each iteration as if it were a separate function call
  for (unsigned i = 0; i < arity; ++i) {
    FunctionType callerTy = getFunctionTypeForIteration(i);

    // attempt unification between the body's formal type and
    // the actual caller type at this iteration
    if (failed(trait::buildSpecializationSubstitution(calleeTy, callerTy, module, err)))
      return failure();
  }

  return success();
}

LogicalResult MapOp::verifySymbolUsesWithUnknownArity() {
  // with unknown arity, all contributing tuple shapes are polymorphic
  // enforce that the body is *purely* polymorphic so it can instantiate later
  FunctionType calleeTy = getBodyFunctionType();

  // every body argument must be purely polymorphic
  for (auto [i, Ei] : llvm::enumerate(calleeTy.getInputs())) {
    if (!trait::isPurelyPolymorphicType(Ei))
      return emitOpError() << "body argument #" << i
                           << " must be purely polymorphic (all leaves are e.g. '!trait.poly'); got "
                           << Ei;
  }

  // the yielded result must also be purely polymorphic
  Type yieldFormal = calleeTy.getResult(0);
  if (!trait::isPurelyPolymorphicType(yieldFormal))
    return emitOpError() << "body yield/result must be purely polymorphic; got "
                         << yieldFormal;

  return success();
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

/// Build the *actual* function type for iteration `elemIdx`.
/// - Concrete tuple inputs contribute their element at this index.
/// - Polymorphic tuple inputs contribute the body’s own formal at that position,
///   so unification sees “actual = formal” (no new binding).
/// - Result is the element type of the op’s result tuple at `elemIdx`.
FunctionType MapOp::getFunctionTypeForIteration(unsigned elemIdx) {
  assert(getArity() && "MapOp::getFunctionTypeForIteration requires known arity");

  SmallVector<Type> argTys;
  FunctionType calleeTy = getBodyFunctionType();

  for (auto [inputIdx, v] : llvm::enumerate(getInputs())) {
    if (auto tt = dyn_cast<TupleType>(v.getType())) {
      argTys.push_back(tt.getType(elemIdx));
    } else {
      // input is !tuple.poly — use the body’s formal at this position
      argTys.push_back(calleeTy.getInput(inputIdx));
    }
  }

  Type resultElemTy = getResultTupleType().getType(elemIdx);
  return FunctionType::get(getContext(), argTys, resultElemTy);
}

llvm::DenseMap<Type,Type> MapOp::buildSubstitutionForIteration(unsigned int i) {
  auto bodyTy = getBodyFunctionType();
  auto iterationTy = getFunctionTypeForIteration(i);

  auto module = getOperation()->getParentOfType<ModuleOp>();
  auto subst = trait::buildSpecializationSubstitution(bodyTy, iterationTy, module);
  if (failed(subst)) {
    llvm_unreachable("MapOp::buildSubstitutionForIteration: unification failed");
  }
  return *subst;
}

}
