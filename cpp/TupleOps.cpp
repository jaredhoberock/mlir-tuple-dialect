// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include "TupleTypes.hpp"
#include <ConstantValue.hpp>
#include <iostream>
#include <mlir/IR/Builders.h>
#include <TraitTypes.hpp>

#define GET_OP_CLASSES
#include <TupleOps.cpp.inc>

namespace mlir::tuple {

static bool isTupleStructureTrait(StringRef traitName) {
  return traitName == "Tuple";
}

/// The type arguments carrying a per-element body's declaration to the types
/// standing opposite it.
///
/// A body is a callable, so its signature is its declaration and the parameters
/// that signature spells are the only variables; the `actuals` -- the types an
/// iteration supplies, the accumulator a step yields, the type the op itself
/// spells -- are rigid, and nothing they spell is narrowed to fit the body.
/// Every position is read before any is compared, so a parameter standing at
/// two of them is fixed at the first and both positions answer to it. The
/// verdict is each formal instantiated at those arguments being its actual,
/// compared as written: an op holding a body reads a spelling through no
/// established context of its own.
///
/// A position whose actual carries no type is one this caller does not name --
/// an input the op cannot index, a yield the op reads off the body rather than
/// dictating. Nothing is read there and nothing is compared.
///
/// A parameter standing opposite a spelling of itself -- the tuple-kinded
/// occurrence of a parameter opposite that same parameter -- takes no argument
/// from that position: the two spell one type, and carrying the parameter to
/// its kinded occurrence would rewrite every bare occurrence in the body into
/// the kinded one when the body is stamped out.
static FailureOr<trait::SpecializationMap> matchBodyPositions(
    FunctionType bodyTy, ArrayRef<Type> formals, ArrayRef<Type> actuals,
    llvm::function_ref<InFlightDiagnostic()> err) {
  assert(formals.size() == actuals.size() &&
         "every formal position stands opposite one actual");

  trait::TypeArguments args(trait::getTypeParametersIn(Type(bodyTy)));
  for (auto [formal, actual] : llvm::zip(formals, actuals))
    if (actual)
      trait::extractTypeArguments(formal, actual, args);

  trait::SpecializationMap arguments;
  for (trait::GenericTypeInterface parameter : args.getParameters()) {
    std::optional<Type> argument = args.lookup(parameter);
    if (argument && trait::getParameterOccurrence(*argument) != parameter)
      arguments.bind(parameter, *argument);
  }

  for (auto [formal, actual] : llvm::zip(formals, actuals))
    if (actual && failed(trait::verifyEqualAfterInstantiation(
                      formal, arguments, actual, /*normalize=*/nullptr, err)))
      return failure();

  return arguments;
}

/// The type arguments carrying `formal`, one spelling from a per-element body's
/// declaration, to `actual`.
static FailureOr<trait::SpecializationMap> matchBodyFormal(
    FunctionType bodyTy, Type formal, Type actual,
    llvm::function_ref<InFlightDiagnostic()> err) {
  return matchBodyPositions(bodyTy, formal, actual, err);
}

/// The type arguments one iteration of a per-element body takes.
///
/// `supplied` stands opposite the body's signature position by position, its
/// arguments first and then what it yields.
static FailureOr<trait::SpecializationMap> matchBodyToIteration(
    FunctionType bodyTy, ArrayRef<Type> supplied,
    llvm::function_ref<InFlightDiagnostic()> err) {
  SmallVector<Type, 4> declared(bodyTy.getInputs());
  declared.append(bodyTy.getResults().begin(), bodyTy.getResults().end());
  return matchBodyPositions(bodyTy, declared, supplied, err);
}

/// Verifies that a downcast from `!trait.poly` to `!tuple.poly` is justified by
/// a tuple-structure claim.
///
/// A tuple structural claim is a proof that the value's polymorphic type
/// variable represents a tuple. The downcast is valid exactly when the result
/// type is the value type with that claimed `!trait.poly` rewritten to the
/// corresponding `!tuple.poly`.
static LogicalResult verifyTupleStructuralDowncast(
    Operation *op, Type valueTy, Type claimTy, Type resultTy,
    function_ref<InFlightDiagnostic()> errFn) {
  auto claimType = cast<trait::ClaimType>(claimTy);
  auto traitApp = claimType.getTraitApplication();
  StringRef traitName = traitApp.getTraitName().getValue();
  if (!isTupleStructureTrait(traitName)) {
    errFn() << "claim must be for a tuple-structure trait";
    return failure();
  }

  auto typeArgs = traitApp.getTypeArgs();
  if (typeArgs.size() != 1) {
    errFn() << "tuple-structure claim must have exactly one type argument";
    return failure();
  }

  Type claimArg = typeArgs[0];

  // A concrete claim already names the final type. There is no polymorphic
  // variable to reinterpret structurally, so the only valid conversion is the
  // identity conversion on that concrete type.
  if (trait::isMonomorphicType(claimArg)) {
    if (valueTy != claimArg) {
      errFn() << "value type must match concrete claim type argument";
      return failure();
    }
    if (resultTy != claimArg) {
      errFn() << "result type must match concrete claim type argument";
      return failure();
    }
    return success();
  }

  auto claimPoly = dyn_cast<trait::PolyType>(claimArg);
  if (!claimPoly) {
    errFn() << "polymorphic tuple-structure claim must name one !trait.poly";
    return failure();
  }

  auto structuralPoly = PolyType::get(op->getContext(), claimPoly);
  llvm::DenseMap<Type, Type> substitution;
  substitution[claimPoly] = structuralPoly;

  // The downcast may only replace the exact polymorphic type named by the
  // claim. If the value type does not contain it, the claim proves nothing
  // about this value.
  Type expected = trait::applySubstitutionOnce(substitution, valueTy);
  if (expected == valueTy) {
    errFn() << "value type does not contain the claimed tuple-polymorphic type";
    return failure();
  }

  // Requiring the caller-provided result type to equal the computed rewrite
  // prevents the op from using a valid tuple claim to smuggle in unrelated
  // structural type changes.
  if (expected != resultTy) {
    errFn() << "result type must be the value type with " << claimPoly
            << " rewritten to " << structuralPoly;
    return failure();
  }

  return success();
}


//===----------------------------------------------------------------------===//
// AllOp
//===----------------------------------------------------------------------===//

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

SmallVector<Type> AllOp::getSuppliedTypesForIteration(unsigned int i) {
  auto inputTupleType = getInputTupleTypeWithKnownArity();
  if (failed(inputTupleType))
    llvm_unreachable("AllOp::getSuppliedTypesForIteration: input must be TupleType");

  // the element this iteration reads; what the body yields is i1 by this op's
  // own rule, checked where the body is
  return {inputTupleType->getType(i), Type()};
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

  // the body's declaration must carry to the element each iteration reads
  for (unsigned i = 0; i < arity; ++i) {
    if (failed(matchBodyToIteration(calleeTy, getSuppliedTypesForIteration(i),
                                    err)))
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


//===----------------------------------------------------------------------===//
// DowncastOp
//===----------------------------------------------------------------------===//

LogicalResult DowncastOp::verify() {
  return verifyTupleStructuralDowncast(
      getOperation(), getValue().getType(), getClaim().getType(),
      getResult().getType(), [&]() { return emitOpError(); });
}

LogicalResult DowncastOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Once the value has been specialized to a concrete tuple type, the
  // structural promise carried by !tuple.poly is no longer needed. A
  // polymorphic value's downcast result is a fresh PolyType, which
  // inference refuses to mint.
  Type valueTy = operands[0].getType();
  if (trait::isPolymorphicType(valueTy))
    return failure();

  inferredReturnTypes.push_back(valueTy);
  return success();
}

LogicalResult DowncastOp::refineReturnTypes(
    MLIRContext *ctx, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  return trait::refineUnlessUnmintable<DowncastOp>(ctx, location, operands, attrs,
                                             properties, regions, returnTypes);
}


//===----------------------------------------------------------------------===//
// AppendOp
//===----------------------------------------------------------------------===//

FailureOr<Type> AppendOp::inferResultType(
    Type tupleTy,
    Type elementTy,
    function_ref<InFlightDiagnostic()> errFn) {
  if (!isTupleLike(tupleTy)) {
    if (errFn) errFn() << "tuple operand must be TupleLike";
    return failure();
  }

  MLIRContext *ctx = tupleTy.getContext();

  // An opaque polymorphic tuple determines no append result: which tuple it
  // stands for is what says what appending to it yields. Construction supplies
  // the result type there; this reads one off the operands or refuses.
  if (trait::isPolymorphicType(tupleTy)) {
    if (errFn) errFn() << "tuple operand must be a concrete tuple type, got "
                       << tupleTy;
    return failure();
  }

  // concrete TupleType case: concatenate element types
  auto concreteTupleTy = dyn_cast<TupleType>(tupleTy);
  if (!concreteTupleTy) {
    if (errFn) errFn() << "tuple operand must be TupleType";
    return failure();
  }

  SmallVector<Type> elems;
  elems.reserve(concreteTupleTy.size() + 1);
  elems.append(concreteTupleTy.begin(), concreteTupleTy.end());
  elems.push_back(elementTy);

  return TupleType::get(ctx, elems);
}

LogicalResult AppendOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // An opaque polymorphic tuple has no determined append result (it would
  // be a fresh PolyType, which inference refuses to mint).
  auto tupleTy = dyn_cast<TupleType>(operands[0].getType());
  if (!tupleTy || trait::isPolymorphicType(operands[1].getType()))
    return failure();

  auto inferred = inferResultType(tupleTy, operands[1].getType());
  if (failed(inferred))
    return failure();

  inferredReturnTypes.push_back(*inferred);
  return success();
}

LogicalResult AppendOp::refineReturnTypes(
    MLIRContext *ctx, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  return trait::refineUnlessUnmintable<AppendOp>(ctx, location, operands, attrs,
                                             properties, regions, returnTypes);
}

LogicalResult AppendOp::verify() {
  // tuple.append has two modes
  // 1. concrete mode: input tuple is tuple, result type must also be TupleType
  // 2. polymorphic mode: input tuple is !tuple.poly, result type must be !tuple.poly
  Type inputTy = getTuple().getType();

  // if input type is tuple<a,b,c>
  if (TupleType concreteTupleTy = dyn_cast<TupleType>(inputTy)) {
    // infer the result type
    auto expectedResultTy = inferResultType(concreteTupleTy, getElement().getType());
    if (failed(expectedResultTy))
      return failure();

    if (*expectedResultTy != getResult().getType())
      return emitOpError() << "type mismatch: expected '" << *expectedResultTy << "'"
                           << "got '" << getResult().getType() << "'";
    return success();
  }

  // if input type is !tuple.poly
  if (PolyType polyTupleTy = dyn_cast<PolyType>(inputTy)) {
    // expected result type is !tuple.poly
    if (!isa<PolyType>(getResult().getType()))
      return emitOpError() << "type mismatch: expected '!tuple.poly', got '"
                           << getResult().getType() << "'";
    return success();
  }

  return emitError() << "unsupported type for input tuple: '" << inputTy << "'";
}


//===----------------------------------------------------------------------===//
// CatOp
//===----------------------------------------------------------------------===//

LogicalResult CatOp::verify() {
  Type lhsTy = getLhs().getType();
  Type rhsTy = getRhs().getType();
  Type resTy = getResult().getType();

  if (!isTupleLike(lhsTy) || !isTupleLike(rhsTy))
    return emitOpError() << "operands must be TupleLike, got "
                         << lhsTy << " and " << rhsTy;

  // result must be TupleLike
  if (!isTupleLike(resTy))
    return emitOpError() << "result must be TupleLike, got "
                         << resTy;

  // if either operand is polymorphic, result must be polymorphic
  if (trait::isPolymorphicType(lhsTy) || trait::isPolymorphicType(rhsTy)) {
    if (!isa<PolyType>(resTy))
      return emitOpError() << "result must be !tuple.poly, got " << resTy;
    return success();
  }

  // concrete case
  auto lhsTT = dyn_cast<TupleType>(lhsTy);
  auto rhsTT = dyn_cast<TupleType>(rhsTy);
  auto resTT = dyn_cast<TupleType>(resTy);

  if (!lhsTT || !rhsTT || !resTT)
    return emitOpError() << "expected concrete operands and result tuple types, got "
                         << lhsTy << ", " << rhsTy << " -> " << resTy;

  // compute expected concatenation
  SmallVector<Type, 8> elems;
  elems.reserve(lhsTT.size() + rhsTT.size());
  elems.append(lhsTT.begin(), lhsTT.end());
  elems.append(rhsTT.begin(), rhsTT.end());
  TupleType expected = TupleType::get(getContext(), elems);

  if (resTT != expected)
    return emitOpError() << "result type must be concatenation of operand element types; "
                         << "expected " << expected << ", got " << resTT;

  return success();
}

FailureOr<Type> CatOp::inferResultType(Type lhsTy, Type rhsTy, llvm::function_ref<InFlightDiagnostic()> errFn) {
  if (!isTupleLike(lhsTy) || !isTupleLike(rhsTy)) {
    if (errFn) errFn() << "operands must be TupleLike";
    return failure();
  }

  MLIRContext *ctx = lhsTy.getContext();

  // An opaque polymorphic operand determines no concatenation: which tuple it
  // stands for is what says how long the result is. Construction supplies the
  // result type there.
  if (trait::isPolymorphicType(lhsTy) || trait::isPolymorphicType(rhsTy)) {
    if (errFn) errFn() << "operands must be concrete tuple types, got " << lhsTy
                       << " and " << rhsTy;
    return failure();
  }

  // concrete TupleType case: concatenate element types
  auto lhs = dyn_cast<TupleType>(lhsTy);
  auto rhs = dyn_cast<TupleType>(rhsTy);
  if (!lhs || !rhs) {
    if (errFn) errFn() << "operands must be TupleTypes";
    return failure();
  }

  SmallVector<Type> elems;
  elems.reserve(lhs.size() + rhs.size());
  elems.append(lhs.begin(), lhs.end());
  elems.append(rhs.begin(), rhs.end());

  return TupleType::get(ctx, elems);
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

  // The claims operand may still be spelled as the mapper's associated type,
  // which contributes no arity until resolution writes the claim tuple.
  if (Value c = getClaims())
    if (!isa<trait::ProjectionType>(c.getType()))
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

  // The formal built here is a declaration of its own -- a pattern whose
  // parameters the operand's spelling is read against -- so its labels start at
  // 0 and are local to it.

  // unknown arity -> expect !tuple.poly
  if (!arity)
    return tuple::PolyType::get(ctx, 0);

  auto LT = dyn_cast<TupleType>(L);
  auto RT = dyn_cast<TupleType>(R);

  // if exactly one side is concrete, the other's elements are this formal's own
  // parameters, one per position
  if (!LT)
    LT = getTupleTypeWithPolymorphicElements(ctx, *arity, /*firstLabel=*/0);
  else if (!RT)
    RT = getTupleTypeWithPolymorphicElements(ctx, *arity, /*firstLabel=*/0);

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

  // A claims operand still spelled as the mapper's associated type names the
  // claims rather than listing them, so what it must equal is the projection
  // this op's own operand types name. The elementwise comparison below runs
  // once resolution writes the claim tuple in the projection's place.
  Type formalClaimsTy;
  if (isa<trait::ProjectionType>(claims.getType())) {
    formalClaimsTy = getMapperClaimsProjection();
  } else {
    auto elementwise = getFormalClaimsTypeForCmpOp(
      getContext(),
      getLhs().getType(),
      getRhs().getType(),
      getTraitRefAttr(),
      getArity(),
      errFn
    );
    if (failed(elementwise)) return failure();
    formalClaimsTy = *elementwise;
  }

  // The claims this comparison requires, read against the operand that supplies
  // them: the parameters the formal spells -- among them the placeholders minted
  // above for a side whose elements this op cannot name -- take what the operand
  // spells opposite them, and the operand's spelling is rigid.
  if (failed(trait::matchDeclaration(trait::getTypeParametersIn(formalClaimsTy),
                                     formalClaimsTy, claims.getType(),
                                     /*normalize=*/nullptr, errFn)))
    return failure();

  return success();
}

Type CmpOp::getMapperClaimsProjection() {
  MLIRContext *ctx = getContext();
  auto mapperApp = trait::TraitApplicationAttr::get(
    ctx,
    FlatSymbolRefAttr::get(ctx, getMapperTraitName(getTraitName())),
    ArrayRef<Type>{getLhs().getType(), getRhs().getType()}
  );
  return trait::ProjectionType::get(ctx, mapperApp,
                                    StringAttr::get(ctx, "Claims"),
                                    /*assocTypeArgs=*/{});
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
// ExclusiveScanOp
//===----------------------------------------------------------------------===//

LogicalResult ExclusiveScanOp::verify() {
  Block &body = getBody().front();
  if (body.getNumArguments() != 2)
    return emitOpError() << "body block must have exactly 2 arguments (accumulator, element), got "
                         << body.getNumArguments();

  if (body.empty())
    return emitOpError("body block cannot be empty");

  if (!isa<YieldOp>(body.back()))
    return emitOpError("body block must terminate with `tuple.yield`, got ")
           << body.back().getName();

  Type inputTy = getInput().getType();
  if (!isTupleLike(inputTy))
    return emitOpError() << "input must be TupleLike, got " << inputTy;

  Type resultTy = getResult().getType();
  if (!isTupleLike(resultTy))
    return emitOpError() << "result must be TupleLike, got " << resultTy;

  // if input is concrete, result arity must be input arity + 1
  if (auto inputTT = dyn_cast<TupleType>(inputTy)) {
    auto resultTT = dyn_cast<TupleType>(resultTy);
    if (!resultTT)
      return emitOpError() << "result must be concrete TupleType when tuple is concrete";
    if (resultTT.size() != inputTT.size() + 1)
      return emitOpError() << "result arity (" << resultTT.size()
                           << ") must be input arity + 1 (" << inputTT.size() + 1 << ")";
  }

  return success();
}

LogicalResult ExclusiveScanOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return emitOpError("not contained in a module");

  if (auto N = getInputArity())
    return verifySymbolUsesWithKnownArity(moduleOp, *N);
  return verifySymbolUsesWithUnknownArity(moduleOp);
}

LogicalResult ExclusiveScanOp::verifySymbolUsesWithKnownArity(ModuleOp module, unsigned arity) {
  auto err = [&]{ return emitOpError(); };

  FunctionType calleeTy = getBodyFunctionType();
  Type AccFormal = calleeTy.getInput(0);
  Type YieldFormal = calleeTy.getResult(0);

  // The body must preserve the accumulator's shape: the next step's accumulator
  // is this step's yield, so the yield is the type in hand and the accumulator
  // formal is the declaration read against it.
  if (failed(matchBodyFormal(calleeTy, AccFormal, YieldFormal, err)))
    return failure();

  // thread accumulator type across iterations, collecting result element types
  SmallVector<Type> resultElemTypes;
  resultElemTypes.reserve(arity + 1);

  Type prev = getInit().getType();
  resultElemTypes.push_back(prev);

  for (unsigned i = 0; i < arity; ++i) {
    auto subst = matchBodyToIteration(
        calleeTy, getSuppliedTypesForIteration(i, prev), err);
    if (failed(subst))
      return failure();

    // what this step yields is what the next one accumulates
    prev = trait::instantiate(YieldFormal, *subst);
    resultElemTypes.push_back(prev);
  }

  // verify result type matches expected
  TupleType expectedResultTy = TupleType::get(getContext(), resultElemTypes);
  if (getResult().getType() != expectedResultTy)
    return err() << "result type mismatch: expected " << expectedResultTy
                 << ", got " << getResult().getType();

  return success();
}

LogicalResult ExclusiveScanOp::verifySymbolUsesWithUnknownArity(ModuleOp module) {
  auto err = [&]{ return emitOpError(); };

  FunctionType calleeTy = getBodyFunctionType();
  Type AccFormal = calleeTy.getInput(0);
  Type ElemFormal = calleeTy.getInput(1);
  Type YieldFormal = calleeTy.getResult(0);
  Type initActual = getInit().getType();

  // The accumulator formal must carry to the init type.
  // This ensures the body can accept the initial value.
  if (failed(matchBodyFormal(calleeTy, AccFormal, initActual, err)))
    return failure();

  // When arity is unknown, we can't verify each element type individually.
  // The element formal must be purely polymorphic (e.g., !trait.a) so it can
  // accept any element type when the tuple is eventually instantiated.
  if (!trait::isPurelyPolymorphicType(ElemFormal))
    return emitOpError() << "element body argument must be purely polymorphic; got " << ElemFormal;

  // The body must preserve the accumulator's shape: the next step's accumulator
  // is this step's yield, so the yield is the type in hand and the accumulator
  // formal is the declaration read against it.
  if (failed(matchBodyFormal(calleeTy, AccFormal, YieldFormal, err)))
    return failure();

  // Result must be polymorphic when input is polymorphic
  if (!isa<PolyType>(getResult().getType()))
      return emitOpError() << "result must be !tuple.poly when input is polymorphic";

  return success();
}

YieldOp ExclusiveScanOp::bodyYield() {
  return cast<YieldOp>(getBody().front().back());
}

FunctionType ExclusiveScanOp::getBodyFunctionType() {
  return FunctionType::get(
      getContext(),
      getBody().front().getArgumentTypes(),
      bodyYield().getOperand().getType()
  );
}

SmallVector<Type> ExclusiveScanOp::getSuppliedTypesForIteration(
    unsigned int i,
    Type accumulatorType) {
  auto tupleType = getInputTupleTypeWithKnownArity();
  if (failed(tupleType))
    llvm_unreachable("getSuppliedTypesForIteration: tuple must be TupleType");

  // the accumulator this step starts from and the element it reads; what the
  // body yields is what the next step accumulates, read off the body
  return {accumulatorType, tupleType->getType(i), Type()};
}

FailureOr<trait::SpecializationMap> ExclusiveScanOp::buildSubstitutionForIteration(
    unsigned int i,
    Type accumulatorType,
    function_ref<InFlightDiagnostic()> errFn) {
  auto tupleType = getInputTupleTypeWithKnownArity(errFn);
  if (failed(tupleType))
    return failure();

  return matchBodyToIteration(getBodyFunctionType(),
                              getSuppliedTypesForIteration(i, accumulatorType),
                              errFn);
}


//===----------------------------------------------------------------------===//
// GetOp
//===----------------------------------------------------------------------===//

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

/// A tuple.get of a tuple.constant folds to the selected element attribute; the
/// folder materializes it at the element type through that type's own dialect.
OpFoldResult GetOp::fold(FoldAdaptor adaptor) {
  auto array = dyn_cast_or_null<ArrayAttr>(adaptor.getTuple());
  if (!array)
    return {};
  return array[getIndex().getSExtValue()];
}


//===----------------------------------------------------------------------===//
// DropLastOp
//===----------------------------------------------------------------------===//

FailureOr<Type> DropLastOp::inferResultType(Type inputTy, function_ref<InFlightDiagnostic()> errFn) {
  if (!isTupleLike(inputTy)) {
    if (errFn) errFn() << "input must be TupleLike, got " << inputTy;
    return failure();
  }

  // An opaque polymorphic tuple determines no prefix: which tuple it stands for
  // is what says what dropping its last element yields. Construction supplies
  // the result type there.
  if (isa<PolyType>(inputTy)) {
    if (errFn) errFn() << "input must be a concrete tuple type, got " << inputTy;
    return failure();
  }

  // concrete case
  auto tupleTy = dyn_cast<TupleType>(inputTy);
  if (!tupleTy) {
    if (errFn) errFn() << "expected concrete TupleType, got " << inputTy;
    return failure();
  }

  if (tupleTy.size() == 0) {
    if (errFn) errFn() << "cannot drop last from empty tuple";
    return failure();
  }

  SmallVector<Type> elems(tupleTy.getTypes().drop_back());
  return TupleType::get(inputTy.getContext(), elems);
}

LogicalResult DropLastOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // An opaque polymorphic tuple has no determined prefix type (it would be
  // a fresh PolyType, which inference refuses to mint).
  Type inputTy = operands[0].getType();
  if (isa<PolyType>(inputTy))
    return failure();

  auto inferred = inferResultType(inputTy);
  if (failed(inferred))
    return failure();

  inferredReturnTypes.push_back(*inferred);
  return success();
}

LogicalResult DropLastOp::refineReturnTypes(
    MLIRContext *ctx, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  return trait::refineUnlessUnmintable<DropLastOp>(ctx, location, operands, attrs,
                                             properties, regions, returnTypes);
}

LogicalResult DropLastOp::verify() {
  Type inputTy = getInput().getType();
  Type resultTy = getResult().getType();

  // polymorphic case
  if (isa<PolyType>(inputTy)) {
    if (!isa<PolyType>(resultTy))
      return emitOpError() << "result must be !tuple.poly when input is polymorphic, got " << resultTy;
    return success();
  }

  // concrete case
  auto expectedResultTy = inferResultType(inputTy, [&]() { return emitOpError(); });
  if (failed(expectedResultTy))
    return failure();

  if (resultTy != *expectedResultTy)
    return emitOpError() << "result type mismatch: expected " << *expectedResultTy
                         << ", got " << resultTy;

  return success();
}


//===----------------------------------------------------------------------===//
// FlattenOp
//===----------------------------------------------------------------------===//

FailureOr<TupleType> FlattenOp::inferKnownArityResultType(Type inputTy, MLIRContext *ctx, function_ref<InFlightDiagnostic()> errFn) {
  auto inputTupleTy = dyn_cast<TupleType>(inputTy);

  // This helper only applies when the *outer* tuple has known arity.
  if (!inputTupleTy) {
    if (errFn)
      errFn() << "input is not a concrete tuple type: " << inputTy;
    return failure();
  }

  SmallVector<Type, 8> flattenedElems;

  for (Type elemTy : inputTupleTy.getTypes()) {
    // Elements must at least be TupleLike.
    if (!isTupleLike(elemTy)) {
      if (errFn)
        errFn() << "all elements of input tuple must be TupleLike, got "
                << elemTy;
      return failure();
    }

    // To infer a *known-arity* result type, each element must itself be a
    // concrete TupleType. If an element is polymorphic TupleLike, overall
    // arity is not known.
    if (auto inner = dyn_cast<TupleType>(elemTy)) {
      flattenedElems.append(inner.begin(), inner.end());
    } else {
      // Some element is polymorphic TupleLike (e.g. !tuple.poly), so we
      // can't infer a fully known-arity result type.
      return failure();
    }
  }

  return TupleType::get(ctx, flattenedElems);
}

LogicalResult FlattenOp::verify() {
  Type inputTy  = getInput().getType();
  Type resultTy = getResult().getType();

  // both input and result must be TupleLike
  if (!isTupleLike(inputTy))
    return emitOpError() << "input must be TupleLike, got " << inputTy;
  if (!isTupleLike(resultTy))
    return emitOpError() << "result must be TupleLike, got " << resultTy;

  // case 1: input is polymorphic (!tuple.poly<X>)
  auto inputPoly = dyn_cast<PolyType>(inputTy);
  if (inputPoly) {
    // result must also be polymorphic
    auto resultPoly = dyn_cast<PolyType>(resultTy);
    if (!resultPoly)
      return emitOpError()
             << "result must be '!tuple.poly' when input is polymorphic; got "
             << resultTy;

    // require that flatten actually changes shape, so polys must differ
    if (inputPoly == resultPoly)
      return emitOpError()
             << "input and result polys must be distinct since flatten "
                "may change tuple shape";

    return success();
  }

  // from here on, input must be a concrete TupleType
  auto inputTupleTy = dyn_cast<TupleType>(inputTy);
  if (!inputTupleTy)
    return emitOpError()
           << "non-polymorphic input must be a concrete tuple type, got "
           << inputTy;

  // try to compute known-arity result type if possible
  auto inferred = inferKnownArityResultType(getInput().getType(), getContext(), [&]() { return emitOpError(); });
  if (succeeded(inferred)) {
    Type expected = *inferred;

    if (resultTy != expected)
      return emitOpError()
             << "result type must be the concatenation of inner tuples; "
             << "expected " << expected << ", got " << resultTy;
    return success();
  }

  // if inference failed, some element is polymorphic TupleLike
  // in that case, result must be polymorphic as well
  if (!isa<PolyType>(resultTy))
    return emitOpError()
           << "result type must be '!tuple.poly' when any input tuple "
              "element is polymorphic; got "
           << resultTy;

  return success();
}

LogicalResult FlattenOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Fails when the outer tuple or any element is polymorphic TupleLike:
  // the flattened arity is unknown there and the result would be a fresh
  // PolyType, which inference refuses to mint.
  auto inferred = inferKnownArityResultType(operands[0].getType(), ctx);
  if (failed(inferred))
    return failure();
  inferredReturnTypes.push_back(*inferred);
  return success();
}

LogicalResult FlattenOp::refineReturnTypes(
    MLIRContext *ctx, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  return trait::refineUnlessUnmintable<FlattenOp>(ctx, location, operands, attrs,
                                             properties, regions, returnTypes);
}


//===----------------------------------------------------------------------===//
// FlatMapOp
//===----------------------------------------------------------------------===//

FailureOr<Type> FlatMapOp::inferIntermediateMapType(function_ref<InFlightDiagnostic()> errFn) {
  MLIRContext *ctx = getContext();
  auto arity = getArity();

  // An opaque polymorphic input names no element positions, so the intermediate
  // map stands for some tuple: a variable of the declaration this op sits in,
  // labelled past every label that declaration already binds so the
  // declaration's own substitution leaves it alone.
  if (!arity)
    return PolyType::get(
        ctx, trait::PolyType::get(ctx,
                                  trait::firstUnusedPolyLabel(getOperation())));

  // degenerate case: tuple<> -> map result is also tuple<>
  if (*arity == 0) return TupleType::get(ctx, {});

  // treat the body as (Eformal) -> Yformal
  FunctionType calleeTy = getBodyFunctionType();
  Type Yformal = calleeTy.getResult(0);

  SmallVector<Type, 8> elementTypes;
  elementTypes.reserve(*arity);

  for (unsigned i = 0; i < *arity; ++i) {
    // specialization for this iteration
    auto subst = buildSubstitutionForIteration(i, errFn);
    if (failed(subst)) return failure();

    // instantiate the body result type for this element
    Type Yactual = subst->apply(Yformal);

    if (!isTupleLike(Yactual)) {
      if (errFn) errFn() << "body yield for element " << i
                         << " must be TupleLike; got " << Yactual;
      return failure();
    }

    // for the intermediate map, each outer element is the per-element yield type
    elementTypes.push_back(Yactual);
  }

  // mapResultType = tuple<Y1,...,Yn>
  return TupleType::get(ctx, elementTypes);
}

LogicalResult FlatMapOp::verify() {
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

  // input, yield, & result must all be tuple-like
  Type inputTy = getInput().getType();
  if (!isTupleLike(inputTy))
    return emitOpError() << "input must be TupleLike, got " << inputTy;

  Type yieldTy = bodyYield().getResult().getType();
  if (!isTupleLike(yieldTy))
    return emitOpError() << "body must yield a TupleLike type, got " << yieldTy;

  Type resultTy = getResult().getType();
  if (!isTupleLike(resultTy))
    return emitOpError() << "result must be TupleLike, got " << yieldTy;

  // if the input is polymorphic, we don't know the arity statically,
  // so the only sensible result type is !tuple.poly
  if (!isa<TupleType>(inputTy)) {
    if (!isa<PolyType>(resultTy))
      return emitOpError() << "result must be '!tuple.poly' when input is polymorphic; got "
                           << resultTy;
  }

  return success();
}

YieldOp FlatMapOp::bodyYield() {
  return cast<YieldOp>(getBody().front().back());
}

FunctionType FlatMapOp::getBodyFunctionType() {
  return FunctionType::get(
      getContext(),
      getBody().front().getArgumentTypes(),
      bodyYield().getOperand().getType()
  );
}

SmallVector<Type> FlatMapOp::getSuppliedTypesForIteration(unsigned int i) {
  auto inputTupleType = getInputTupleTypeWithKnownArity();
  if (failed(inputTupleType))
    llvm_unreachable("FlatMapOp::getSuppliedTypesForIteration: input must be TupleType");

  // the element this iteration reads; what the body yields is the tuple this
  // element contributes to the result, read off the body
  return {inputTupleType->getType(i), Type()};
}

FailureOr<trait::SpecializationMap> FlatMapOp::buildSubstitutionForIteration(
    unsigned int i,
    function_ref<InFlightDiagnostic()> errFn) {
  return matchBodyToIteration(getBodyFunctionType(),
                              getSuppliedTypesForIteration(i), errFn);
}

LogicalResult FlatMapOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // must be inside a module
  auto moduleOp = getOperation()->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return emitOpError("not contained in a module");

  if (auto N = getArity())
    return verifySymbolUsesWithKnownArity(moduleOp, *N);
  return verifySymbolUsesWithUnknownArity(moduleOp);
}

LogicalResult FlatMapOp::verifySymbolUsesWithKnownArity(ModuleOp module, unsigned arity) {
  auto err = [&]{ return emitOpError(); };

  // treat the body as a function: (Eformal) -> Yformal
  FunctionType calleeTy = getBodyFunctionType();
  Type Yformal = calleeTy.getResult(0);

  SmallVector<Type,8> concatenatedElems;
  bool allConcrete = true;

  // for each tuple element i, the body reads that element
  for (unsigned i = 0; i < arity; ++i) {
    auto subst =
        matchBodyToIteration(calleeTy, getSuppliedTypesForIteration(i), err);
    if (failed(subst))
      return failure();

    // instantiate the yield type for this iteration
    Type Yactual = trait::instantiate(Yformal, *subst);

    // each iteration must yield something TupleLike
    if (!isTupleLike(Yactual))
      return err() << "body yield for element " << i
                   << " must be TupleLike, got " << Yactual;

    // try to refine to a concrete TupleType
    if (auto tt = dyn_cast<TupleType>(Yactual)) {
      concatenatedElems.append(tt.begin(), tt.end());
    } else {
      // we hit a polymorphic tuple::PolyType; we can
      // no longer compute a concrete concatenation.
      allConcrete = false;
    }
  }

  Type resultTy = getResult().getType();

  if (allConcrete) {
    // we know every yield's arity -> result must be the exact concatentation
    Type expectedResTy = TupleType::get(getContext(), concatenatedElems);
    if (resultTy != expectedResTy)
      return err() << "result type must be concatenation of yielded tuples; "
                   << "expected " << expectedResTy << ", got " << resultTy;
    return success();
  }

  // at least one yield is polymorphic: we've already checked that the result is TupleLike,
  // and that the body is specialization-safe. let higher-level dialects
  // enforce stronger shape invariants if they want
  return success();
}

LogicalResult FlatMapOp::verifySymbolUsesWithUnknownArity(ModuleOp module) {
  // with unknown arity, the input is polymorphic (!tuple.poly). we require the
  // body to be purely polymorphic so it can instantiate later for any arity.
  FunctionType calleeTy = getBodyFunctionType();
  Type Eformal = calleeTy.getInput(0);
  Type Yformal = calleeTy.getResult(0);

  if (!trait::isPurelyPolymorphicType(Eformal))
    return emitOpError()
           << "body argument must be purely polymorphic (all leaves e.g. '!trait.poly'); got "
           << Eformal;

  if (!isTupleLike(Yformal))
    return emitOpError()
           << "body yield/result must be TupleLike; got "
           << Yformal;

  if (!trait::isPurelyPolymorphicType(Yformal))
    return emitOpError()
           << "body yield/result must be purely polymorphic; got "
           << Yformal;

  return success();
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
    // read each iteration in isolation as if it were a separate call
    auto subst =
        matchBodyToIteration(calleeTy, getSuppliedTypesForIteration(i, prev), err);
    if (failed(subst)) return failure();

    // what this step yields is what the next one accumulates
    prev = trait::instantiate(R, *subst);
  }

  // The fold's result is the type the last step yielded.
  return trait::verifyEqualAfterInstantiation(prev, trait::SpecializationMap(),
                                              getResult().getType(),
                                              /*normalize=*/nullptr, err);
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

  // the accumulator formal must carry to the actual init type
  if (failed(matchBodyFormal(calleeTy, accFormal, initActual, err)))
    return failure();

  // every non-accumulator body arg type must be purely polymorphic
  for (auto [i, Ei] : llvm::enumerate(calleeTy.getInputs().drop_front())) {
    if (!trait::isPurelyPolymorphicType(Ei))
      return err() << "non-accumulator body argument #" << (i + 1)
                   << " must be purely polymorphic (all leaves are e.g. '!trait.poly'); got "
                   << Ei;
  }

  // Closure: one step of the body must preserve the accumulator's shape. The
  // next step's accumulator is this step's yield, so the yield is the type in
  // hand and the accumulator formal is the declaration read against it.
  if (failed(matchBodyFormal(calleeTy, accFormal, yieldFormal, err)))
    return failure();

  // op result consistency: the op's own result type is the term the accumulator
  // formal must carry to
  return matchBodyFormal(calleeTy, accFormal, resultFormal, err);
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

SmallVector<Type> FoldlOp::getSuppliedTypesForIteration(
    unsigned int i,
    Type resultTypeOfPreviousIteration) {
  auto inputTupleTypes = getInputTypesAsTupleTypes();
  if (failed(inputTupleTypes))
    llvm_unreachable("FoldlOp::getSuppliedTypesForIteration: inputs must be TupleTypes");

  // the accumulator this step starts from and the element each input
  // contributes; what the body yields is what the next step accumulates, read
  // off the body
  SmallVector<Type> supplied;
  supplied.push_back(resultTypeOfPreviousIteration);
  for (TupleType input : *inputTupleTypes)
    supplied.push_back(input.getType(i));
  supplied.push_back(Type());

  return supplied;
}

trait::SpecializationMap FoldlOp::buildSubstitutionForIteration(
    unsigned int i, 
    Type resultTypeOfPreviousIteration) {
  assert(inputTypesAreTupleTypes() && "FoldlOp::buildSubstitutionForIteration: inputs must be TupleType");

  auto subst = matchBodyToIteration(
      getBodyFunctionType(),
      getSuppliedTypesForIteration(i, resultTypeOfPreviousIteration),
      /*err=*/nullptr);
  if (failed(subst)) {
    // this should never happen if FoldlOp::verifySymbolUses succeeds
    llvm_unreachable("buildSubstitutionForIteration: the body does not take this iteration");
  }
  return *subst;
}


//===----------------------------------------------------------------------===//
// LastOp
//===----------------------------------------------------------------------===//

FailureOr<Type> LastOp::inferResultType(Type inputTy, function_ref<InFlightDiagnostic()> errFn) {
  if (!isTupleLike(inputTy)) {
    if (errFn) errFn() << "input must be TupleLike, got " << inputTy;
    return failure();
  }

  // An opaque polymorphic tuple determines no last element: which tuple it
  // stands for is what says what its last element is. Construction supplies the
  // result type there.
  if (isa<PolyType>(inputTy)) {
    if (errFn) errFn() << "input must be a concrete tuple type, got " << inputTy;
    return failure();
  }

  // concrete case
  auto tupleTy = dyn_cast<TupleType>(inputTy);
  if (!tupleTy) {
    if (errFn) errFn() << "expected TupleType, got " << inputTy;
    return failure();
  }

  if (tupleTy.size() == 0) {
    if (errFn) errFn() << "cannot get last element of empty tuple";
    return failure();
  }

  return tupleTy.getType(tupleTy.size() - 1);
}

LogicalResult LastOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Inference is deterministic: it fails rather than mint a fresh
  // PolyType. An opaque polymorphic tuple has no determined element type,
  // so that case is construction's job (pass an explicit result type).
  Type inputTy = operands[0].getType();
  if (isa<PolyType>(inputTy))
    return failure();

  auto inferred = inferResultType(inputTy);
  if (failed(inferred))
    return failure();

  inferredReturnTypes.push_back(*inferred);
  return success();
}

LogicalResult LastOp::refineReturnTypes(
    MLIRContext *ctx, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  return trait::refineUnlessUnmintable<LastOp>(ctx, location, operands, attrs,
                                             properties, regions, returnTypes);
}

LogicalResult LastOp::verify() {
  Type inputTy = getInput().getType();
  Type resultTy = getResult().getType();

  // polymorphic case
  if (isa<PolyType>(inputTy)) {
    if (!isa<trait::PolyType>(resultTy))
      return emitOpError() << "result must be !trait.poly when input is polymorphic, got " << resultTy;
    return failure();
  }

  // concrete case
  auto expectedResultTy = inferResultType(inputTy, [&]() { return emitOpError(); });
  if (failed(expectedResultTy))
    return failure();

  if (resultTy != *expectedResultTy)
    return emitOpError() << "result type mismatch: expected " << *expectedResultTy
                         << ", got " << resultTy;

  return success();
}


//===----------------------------------------------------------------------===//
// MakeOp
//===----------------------------------------------------------------------===//

LogicalResult MakeOp::verify() {
  auto tupleTy = dyn_cast<TupleType>(getResult().getType());
  if (!tupleTy)
    return emitOpError("result must be a tuple type");

  if (tupleTy.size() != getNumOperands())
    return emitOpError("operand/result arity mismatch");

  for (auto [i, operand] : llvm::enumerate(getOperands())) {
    Type elemTy = tupleTy.getType(i);
    if (elemTy != operand.getType())
      return emitOpError()
             << "operand " << i << " has type " << operand.getType()
             << ", but result element is " << elemTy;
  }

  return success();
}

LogicalResult MakeOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr, OpaqueProperties, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Total: the result is the tuple of the operand types, determined even
  // when operands mention polymorphism -- no fresh PolyType is needed.
  SmallVector<Type> elementTypes;
  elementTypes.reserve(operands.size());
  for (Value e : operands)
    elementTypes.push_back(e.getType());
  inferredReturnTypes.push_back(TupleType::get(ctx, elementTypes));
  return success();
}

LogicalResult MakeOp::refineReturnTypes(
    MLIRContext *ctx, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &returnTypes) {
  return trait::refineUnlessUnmintable<MakeOp>(ctx, location, operands, attrs,
                                             properties, regions, returnTypes);
}


//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// Checks that `value` describes a constant of the tuple `type`, its scalar and
/// tuple leaves through the shared shape check. A leaf of any other type -- a
/// nominal wrapper, an LLVM struct -- carries that type's own constant attribute,
/// which this local check, with no symbol table, leaves to that type's constant op.
static LogicalResult
verifyConstantValue(Attribute value, Type type,
                    llvm::function_ref<InFlightDiagnostic()> errFn) {
  auto leafHook = [&](Attribute, Type) -> LogicalResult { return success(); };
  return lowering::verifyConstantShape(value, type, errFn, leafHook);
}

LogicalResult ConstantOp::verify() {
  auto tupleTy = cast<TupleType>(getResult().getType());
  return verifyConstantValue(getValue(), tupleTy,
                             [&]() { return emitOpError(); });
}

OpFoldResult ConstantOp::fold(FoldAdaptor) { return getValue(); }


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
    if (failed(matchBodyToIteration(calleeTy, getSuppliedTypesForIteration(i),
                                    err)))
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

/// What iteration `elemIdx` supplies to the body, by position in its signature.
/// - A concrete input contributes its element at this index.
/// - An input whose arity this op does not know names no element there, so it
///   supplies no type and the position carries none.
/// - What the body yields is the element the op’s own result tuple spells at
///   this index.
SmallVector<Type> MapOp::getSuppliedTypesForIteration(unsigned elemIdx) {
  assert(getArity() && "MapOp::getSuppliedTypesForIteration requires known arity");

  SmallVector<Type> supplied;
  for (Value input : getInputs()) {
    auto tt = dyn_cast<TupleType>(input.getType());
    supplied.push_back(tt ? tt.getType(elemIdx) : Type());
  }
  supplied.push_back(getResultTupleType().getType(elemIdx));

  return supplied;
}

FailureOr<trait::SpecializationMap> MapOp::buildSubstitutionForIteration(
    unsigned int i,
    function_ref<InFlightDiagnostic()> errFn) {
  return matchBodyToIteration(getBodyFunctionType(),
                              getSuppliedTypesForIteration(i), errFn);
}

}
