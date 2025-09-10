#include "Canonicalization.hpp"
#include "Dialect.hpp"
#include "Monomorphization.hpp"
#include "Ops.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Transforms/DialectConversion.h>
#include <Instantiation.hpp>
#include <Trait.hpp>

namespace mlir::tuple {

//===----------------------------------------------------------------------===//
// populateConvertTupleToTraitPatterns
//===----------------------------------------------------------------------===//

// synthesizes a trait.trait @tuple.Map<mapped-trait-name> trait if it does not already exist
// this is anchored on the TraitOp whose mapper we want to introduce because we can't
// anchor on ModuleOp
struct IntroduceMapperTrait : OpRewritePattern<trait::TraitOp> {
  StringRef mappedTraitName;

  IntroduceMapperTrait(MLIRContext *ctx, StringRef mappedTraitName)
    : OpRewritePattern<trait::TraitOp>(ctx), mappedTraitName(mappedTraitName) {}

  LogicalResult matchAndRewrite(trait::TraitOp op,
                                PatternRewriter& rewriter) const override {
    // only anchor on the named mapped trait
    if (op.getSymName() != mappedTraitName)
      return rewriter.notifyMatchFailure(op, "not the trait to map");

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "not in a module");

    MLIRContext *ctx = op.getContext();
    std::string name = Twine("tuple.Map" + mappedTraitName).str();

    // if a trait by this name already exists, bail
    if (SymbolTable::lookupNearestSymbolFrom<trait::TraitOp>(
          module, FlatSymbolRefAttr::get(ctx, name)))
      return rewriter.notifyMatchFailure(op, "mapper trait already exists");

    // insert at the end of the module body
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    Location loc = rewriter.getUnknownLoc();

    // create:
    //
    // !S = trait.poly<fresh>
    // !O = trait.poly<fresh>
    // !C = trait.poly<fresh>
    // trait.trait @tuple.Map<mapped-trait-name>[!S,!O,!C] attributes {
    //   tuple.impl_generator = "map",
    //   tuple.mapped_trait = @<mapped-trait-name>
    // } {
    //   func.func private @claims() -> !C
    // }

    Type S = trait::PolyType::fresh(ctx);
    Type O = trait::PolyType::fresh(ctx);
    Type C = trait::PolyType::fresh(ctx);
    
    auto trait = rewriter.create<trait::TraitOp>(
      loc,
      StringAttr::get(ctx, name),
      /*typeParams=*/ArrayRef{S, O, C},
      /*requirements=*/trait::ConstraintsAttr::get(ctx, {})
    );

    // attach attributes for the map generator
    trait->setAttr("tuple.impl_generator", StringAttr::get(ctx, "map"));
    trait->setAttr("tuple.mapped_trait", FlatSymbolRefAttr::get(ctx, mappedTraitName));

    // add @claims() to the trait body
    {
      Block &body = trait.getBody().front();
      rewriter.setInsertionPointToStart(&body);

      auto claimsTy = rewriter.getFunctionType(
        /*inputs=*/TypeRange{},
        /*results=*/C
      );

      auto claimsFn = rewriter.create<func::FuncOp>(
        loc,
        "claims",
        claimsTy
      );
      claimsFn.setPrivate();
    }

    return success();
  }
};

// rewrites tuple.cmp with monomorphic lhs & rhs, but no claims operand
// synthesizes a tuple of claims and then re-emits a tuple.cmp op with
// that additional claims operand
struct CmpOpMonoSynthesizeClaims : OpRewritePattern<CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter& rewriter) const override {
    // only apply when there is no claims operand
    if (op.getClaims())
      return rewriter.notifyMatchFailure(op, "claims already present");

    // only apply when lhs & rhs are monomorphic
    auto inputTupleTypes = op.getMonomorphicTupleOperandTypes();
    if (failed(inputTupleTypes))
      return rewriter.notifyMatchFailure(op, "operands are polymorphic");

    auto [L,R] = *inputTupleTypes;

    MLIRContext* ctx = op.getContext();
    Location loc = op.getLoc();

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    // synthesize per-element claims of the base trait:
    // claim_i = trait.allege @Trait[Li,Ri]
    SmallVector<Value> claimElems;
    FlatSymbolRefAttr traitRef = op.getTraitRefAttr();
    for (auto [Li,Ri] : llvm::zip(L.getTypes(), R.getTypes())) {
      auto app = trait::TraitApplicationAttr::get(ctx, traitRef, {Li,Ri});

      // %ci = trait.allege @Trait[Li,Ri]
      Value ci = rewriter.create<trait::AllegeOp>(loc, app);
      claimElems.push_back(ci);
    }

    // %claims = tuple.make (c1..ck)
    Value claimsTuple = rewriter.create<MakeOp>(loc,claimElems);

    // re-emit the tuple.cmp op with the new claims operand
    rewriter.replaceOpWithNewOp<CmpOp>(
      op,
      op.getPredicate(),
      op.getLhs(),
      op.getRhs(),
      claimsTuple
    );

    return success();
  }
};

// rewrites a tuple.cmp with eq/ne, lhs, rhs, and claims operands
// into a tuple.foldl op
struct CmpOpPartialEqLowering : OpRewritePattern<CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter& rewriter) const override {
    // only handle a PartialEq method
    if (op.getTraitName() != "PartialEq")
      return rewriter.notifyMatchFailure(op, "not PartialEq");
    StringRef method = op.getMethodName();
    if (method != "eq" && method != "ne")
      return rewriter.notifyMatchFailure(op, "only eq/ne supported");

    // only apply when there is a claims operand
    Value claims = op.getClaims();
    if (!claims)
      return rewriter.notifyMatchFailure(op, "claims operand required");

    // handle empty tuples directly so that we do not attempt to call a method below
    if (auto arity = op.getArity(); arity && *arity == 0) {
      const bool val = method == "eq";
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, rewriter.getBoolAttr(val));
      return success();
    }

    Location loc = op.getLoc();
    MLIRContext* ctx = op.getContext();
    auto i1Ty = rewriter.getI1Type();

    // init accumulator: true for eq, false for ne
    Value init = rewriter.create<arith::ConstantOp>(
      loc,
      rewriter.getBoolAttr(method == "eq")
    );

    SmallVector<Value,3> inputs{op.getLhs(), op.getRhs(), claims};
    auto fold = rewriter.create<FoldlOp>(loc, i1Ty, init, inputs);

    // build the body:
    //
    // ^bb0(%acc: i1, %li: !Li, %ri: !Ri, %ci: !trait.claim<@PartialEq[!Li,!Ri]>):
    //   %call_res = trait.method.call %ci @PartialEq[!Li,!Ri]::method(%li, %ri) : ...
    //   %resi = AND or OR with %acc
    //   yield %resi : i1
    {
      Type Li = trait::PolyType::fresh(ctx);
      Type Ri = trait::PolyType::fresh(ctx);
      Type Ci = trait::ClaimType::get(ctx, op.getTraitRefAttr(), {Li,Ri});

      Block *body = rewriter.createBlock(&fold.getBody());
      body->addArguments({i1Ty, Li, Ri, Ci},
                         {loc, loc, loc, loc});

      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(body);

      Value acc = body->getArgument(0);
      Value li = body->getArgument(1);
      Value ri = body->getArgument(2);
      Value ci = body->getArgument(3);

      // call the method requested by the tuple.cmp
      Value callRes = rewriter.create<trait::MethodCallOp>(
        loc, i1Ty,
        op.getTraitName(),
        method,
        ci,
        ValueRange{li,ri}
      ).getResult(0);

      // fold: eq -> AND; ne -> OR
      Value resi = (method == "eq")
        ? (Value)rewriter.create<arith::AndIOp>(loc, acc, callRes)
        : (Value)rewriter.create<arith::OrIOp>(loc, acc, callRes);

      rewriter.create<YieldOp>(loc, resi);
    }

    rewriter.replaceOp(op, fold.getResult());
    return success();
  }
};

// rewrites a tuple.cmp with le/lt/ge/gt, lhs, rhs, and claims operands
// into a tuple.foldl op
struct CmpOpPartialOrdLowering : OpRewritePattern<CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter& rewriter) const override {
    // only handle a PartialOrd method
    if (op.getTraitName() != "PartialOrd")
      return rewriter.notifyMatchFailure(op, "not PartialOrd");

    // only handle lt/le/gt/ge
    const auto pred = op.getPredicate();
    const bool isLtLe   = (pred == CmpPredicate::lt || pred == CmpPredicate::le);
    const bool isStrict = (pred == CmpPredicate::lt || pred == CmpPredicate::gt);
    if (!(pred == CmpPredicate::lt || pred == CmpPredicate::le ||
          pred == CmpPredicate::gt || pred == CmpPredicate::ge))
      return rewriter.notifyMatchFailure(op, "predicate not in lt/le/gt/ge");

    // only apply when there is a claims operand
    Value claims = op.getClaims();
    if (!claims)
      return rewriter.notifyMatchFailure(op, "claims operand required");

    // handle empty tuples directly so that we do not attempt to call a method below
    if (auto arity = op.getArity(); arity && *arity == 0) {
      const bool val = (pred == CmpPredicate::le) || (pred == CmpPredicate::ge);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, rewriter.getBoolAttr(val));
      return success();
    }

    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();

    Type i1 = rewriter.getI1Type();
    TupleType accTy = TupleType::get(ctx, {i1, i1}); // (res, allEq)

    // init: (res=false, allEq=true)
    Value cFalse = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value cTrue  = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    Value init   = rewriter.create<MakeOp>(loc, accTy, ValueRange{cFalse, cTrue});

    // build foldl over (lhs, rhs, claims)
    SmallVector<Value,3> inputs{op.getLhs(), op.getRhs(), claims};
    auto fold = rewriter.create<FoldlOp>(loc, accTy, init, inputs);

    // Body:
    //
    // ^bb0(%acc: tuple<i1,i1>, %li: !Li, %ri: !Ri,
    //      %ci: !trait.claim<@PartialOrd[!Li,!Ri]>):
    //   // Unpack accumulator
    //   %res     = tuple.get %acc, 0 : tuple<i1,i1> -> i1
    //   %all_eq  = tuple.get %acc, 1 : tuple<i1,i1> -> i1
    //
    //   // Per-element calls (exactly two)
    //   %lt_i = trait.method.call %ci @PartialOrd[!Li,!Ri]::@lt(%li, %ri)
    //           : (!Li, !Ri) -> i1
    //   %gt_i = trait.method.call %ci @PartialOrd[!Li,!Ri]::@gt(%li, %ri)
    //           : (!Li, !Ri) -> i1
    //
    //   // Equality at this position
    //   %either = arith.ori %lt_i, %gt_i
    //   %eq_i = arith.xori %either, %true
    //
    //   // Strictness at this position
    //   %strict = (isLtLe ? %lt_i: %gt_i)
    //   
    //   // Contribute only if all previous elements were equal
    //   %contrib = arith.andi %all_eq, %strict : i1
    //
    //   // Accumulate result for strict predicates (lt/gt)
    //   %res_next' = arith.ori %res, %contrib : i1
    //
    //   // Track whether we've stayed equal up to this element
    //   %all_eq_next' = arith.andi %all_eq, %eq_i : i1
    //
    //   // Pack and yield the new accumulator
    //   %acc_next' = tuple.make(%res_next', %all_eq_next') : tuple<i1,i1>
    //   tuple.yield %acc_next' : tuple<i1,i1>
    {
      PatternRewriter::InsertionGuard guard(rewriter);

      Type Li = trait::PolyType::fresh(ctx);
      Type Ri = trait::PolyType::fresh(ctx);
      Type Ci = trait::ClaimType::get(ctx, op.getTraitRefAttr(), {Li, Ri});

      Block *body = rewriter.createBlock(&fold.getBody());
      body->addArguments({accTy, Li, Ri, Ci}, {loc, loc, loc, loc});

      Value acc = body->getArgument(0);
      Value li = body->getArgument(1);
      Value ri = body->getArgument(2);
      Value ci = body->getArgument(3);

      Value res   = rewriter.create<GetOp>(loc, i1, acc, rewriter.getIndexAttr(0));
      Value allEq = rewriter.create<GetOp>(loc, i1, acc, rewriter.getIndexAttr(1));

      Value lt_i = rewriter.create<trait::MethodCallOp>(
        loc, i1, "PartialOrd", "lt", ci, ValueRange{li, ri}).getResult(0);
      Value gt_i = rewriter.create<trait::MethodCallOp>(
        loc, i1, "PartialOrd", "gt", ci, ValueRange{li, ri}).getResult(0);

      Value either = rewriter.create<arith::OrIOp>(loc, lt_i, gt_i);
      Value eq_i   = rewriter.create<arith::XOrIOp>(loc, either, cTrue);

      Value strict = isLtLe ? lt_i : gt_i;

      Value contrib     = rewriter.create<arith::AndIOp>(loc, allEq, strict);
      Value res_next    = rewriter.create<arith::OrIOp>(loc, res, contrib);
      Value all_eq_next = rewriter.create<arith::AndIOp>(loc, allEq, eq_i);

      Value acc_next = rewriter.create<MakeOp>(loc, accTy, ValueRange{res_next, all_eq_next});
      rewriter.create<YieldOp>(loc, acc_next);
    }

    // finalize:
    // lt/gt -> finalRes = res
    // le/ge -> finalRes = res || allEq
    Value finalRes = rewriter.create<GetOp>(loc, i1, fold.getResult(), rewriter.getIndexAttr(0));
    if (!isStrict) {
      Value finalAllEq = rewriter.create<GetOp>(loc, i1, fold.getResult(), rewriter.getIndexAttr(1));
      finalRes = rewriter.create<arith::OrIOp>(loc, finalRes, finalAllEq);
    }

    rewriter.replaceOp(op, finalRes);
    return success();
  }
};

void populateConvertTupleToTraitPatterns(RewritePatternSet& patterns) {
  // introduce the @tuple.MapPartialEq trait
  patterns.add<IntroduceMapperTrait>(patterns.getContext(), "PartialEq");

  // introduce the @tuple.MapPartialOrd trait
  patterns.add<IntroduceMapperTrait>(patterns.getContext(), "PartialOrd");

  patterns.add<
    CmpOpMonoSynthesizeClaims,
    CmpOpPartialEqLowering,
    CmpOpPartialOrdLowering
  >(patterns.getContext());
}


//===----------------------------------------------------------------------===//
// populateInstantiateMonomorphsPatterns
//===----------------------------------------------------------------------===//

struct FoldlOpInstantiation : public OpRewritePattern<FoldlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FoldlOp op,
                                PatternRewriter& rewriter) const override {
    // we can't lower without TupleType inputs
    if (failed(op.getInputTypesAsTupleTypes()))
      return rewriter.notifyMatchFailure(op, "inputs are not all TupleType");

    // we can't lower without a known arity
    auto maybeArity = op.getArity();
    if (!maybeArity) return rewriter.notifyMatchFailure(op, "arity is still unknown");

    unsigned arity = *maybeArity;

    Location loc = op.getLoc();
    Region &body = op.getBody();

    Value previousResult = op.getInit();
    for (unsigned int i = 0; i < arity; ++i) {
      // create a vector of arguments to "pass" to the block below
      SmallVector<Value> args;
      args.push_back(previousResult);

      // collect the ith element from each input tuple
      for (Value tuple : op.getInputs())
        args.push_back(rewriter.create<GetOp>(loc, tuple, i));

      // build the type substitution for this iteration
      DenseMap<Type,Type> subst = op.buildSubstitutionForIteration(i, previousResult.getType());

      // instantiate the body into a temporary Region
      Region bodyInstance;
      trait::instantiatePolymorphicRegion(rewriter, body, bodyInstance, subst);

      // get the block to inline
      Block* block = &bodyInstance.front();

      // before inlining, find the block's YieldOp
      auto yieldOp = cast<YieldOp>(block->getTerminator());

      // inline the block, replacing block arguments with arguments collected above
      rewriter.inlineBlockBefore(block, op, args);

      // now that the yield op has been inlined, grab its operand
      previousResult = yieldOp.getOperand();

      // erase the yield
      rewriter.eraseOp(yieldOp);
    }

    // replace the foldl op with the result of the final iteration
    rewriter.replaceOp(op, previousResult);
    return success();
  }
};

struct MapOpInstantiation : public OpRewritePattern<MapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp op,
                                PatternRewriter& rewriter) const override {
    // we can't lower without TupleType inputs
    if (failed(op.getInputTypesAsTupleTypes()))
      return rewriter.notifyMatchFailure(op, "inputs are not all TupleType");

    // we can't lower without a known arity
    auto maybeArity = op.getArity();
    if (!maybeArity) return rewriter.notifyMatchFailure(op, "arity is still unknown");

    unsigned arity = *maybeArity;

    Location loc = op.getLoc();
    Region &body = op.getBody();

    SmallVector<Value> resultElems;
    resultElems.reserve(arity);

    for (unsigned i = 0; i < arity; ++i) {
      // collect the ith element from each input tuple to pass to the block below
      SmallVector<Value> args;
      args.reserve(op.getInputs().size());
      for (Value tup : op.getInputs()) {
        args.push_back(rewriter.create<GetOp>(loc, tup, i));
      }

      // build the type substitution for this iteration
      auto subst = op.buildSubstitutionForIteration(i);

      // instantiate the body into a temporary Region
      Region bodyInstance;
      trait::instantiatePolymorphicRegion(rewriter, body, bodyInstance, subst);

      // get the block to inline and its yield
      Block* block = &bodyInstance.front();
      auto yieldOp = cast<YieldOp>(block->getTerminator());

      // inline the instantiated block before the map op, binding its arguments
      rewriter.inlineBlockBefore(block, op, args);

      // the yielded value is the ith element of the result tuple
      resultElems.push_back(yieldOp.getOperand());

      // the original yield is now inlined; erase it
      rewriter.eraseOp(yieldOp);
    }

    // replace the map op with the assembled result tuple
    rewriter.replaceOpWithNewOp<MakeOp>(op, resultElems);
    return success();
  }
};

void populateInstantiateMonomorphsPatterns(RewritePatternSet& patterns) {
  // instantiating monomorphs may generate new tuple.cmp ops,
  // so add their patterns to the set
  patterns.add<
    CmpOpPartialEqLowering,
    CmpOpPartialOrdLowering
  >(patterns.getContext());

  // tuple.foldl and tuple.map lowerings instantiate polymorphic regions,
  // so add their patterns to the set
  patterns.add<
    FoldlOpInstantiation,
    MapOpInstantiation
  >(patterns.getContext());
}


//===----------------------------------------------------------------------===//
// populateEraseClaimsPatterns
//===----------------------------------------------------------------------===//

struct EraseClaimsFromGetOp : OpConversionPattern<GetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(GetOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto *tc = getTypeConverter();
    if (!tc)
      return rewriter.notifyMatchFailure(op, "missing TypeConverter");

    // get the old result type
    Type oldResultTy = op.getResult().getType();

    // if the result type is fully erased, erase the op entirely
    SmallVector<Type,2> resultPieces;
    if (failed(tc->convertType(oldResultTy, resultPieces)))
      return rewriter.notifyMatchFailure(op, "result element type conversion failed");
    if (resultPieces.empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    if (resultPieces.size() != 1)
      return rewriter.notifyMatchFailure(op, "result type expanded under conversion");

    // compute the new index: count how many prior elements survive
    unsigned oldIdx = op.getIndexAttr().getInt();
    unsigned newIdx = 0;
    for (auto [i, elemTy] : llvm::enumerate(op.getTupleType().getTypes())) {
      SmallVector<Type,2> pieces;
      if (failed(tc->convertType(elemTy, pieces)))
        return rewriter.notifyMatchFailure(op, "element type conversion failed");
      if (i == oldIdx) {
        // if the target element was erased, we would have returned above
        break;
      }
      newIdx += pieces.size();
    }

    // replace with converted operand and new index
    rewriter.replaceOpWithNewOp<GetOp>(
      op,
      adaptor.getTuple(),
      newIdx
    );

    return success();
  }
};

// EraseClaimsFromMakeOp is a generic conversion pattern instead of a OpConversionPattern<MakeOp>
// because it needs to handle 1:0 type conversions (because !trait.claim types get erased completely)
// and therefore its operands (if they are claim values) may get erased during conversion
struct EraseClaimsFromMakeOp : ConversionPattern {
  EraseClaimsFromMakeOp(TypeConverter &tc, MLIRContext* ctx)
    : ConversionPattern(tc, MakeOp::getOperationName(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                ArrayRef<ValueRange> newOperandRanges,
                                ConversionPatternRewriter &rewriter) const override {
    auto make = dyn_cast<MakeOp>(op);
    if (!make)
      return rewriter.notifyMatchFailure(op, "not a tuple.make");

    auto *tc = getTypeConverter();
    if (!tc)
      return rewriter.notifyMatchFailure(op, "missing TypeConverter");

    // convert the result type
    Type oldResTy = make.getResult().getType();
    Type newResTy = tc->convertType(oldResTy);
    if (!newResTy)
      return rewriter.notifyMatchFailure(op, "result type conversion failed");

    // flatten the N-ary converted operand ranges into a single ValueRange
    SmallVector<Value,8> flatOps;
    for (ValueRange vr : newOperandRanges)
      flatOps.append(vr.begin(), vr.end());

    // if nothing changed (type & operands), we can skip this operation
    bool sameType = (newResTy == oldResTy);
    bool sameOperands = llvm::equal(flatOps, op->getOperands());
    if (sameType && sameOperands)
      return rewriter.notifyMatchFailure(op, "no change");

    // rebuild with the converted operands
    rewriter.replaceOpWithNewOp<MakeOp>(
      op,
      newResTy,
      flatOps
    );

    return success();
  }
};

void populateEraseClaimsPatterns(TypeConverter& converter, RewritePatternSet& patterns) {
  // add a type conversion that recursively applies the converter to TupleType
  converter.addConversion([&](TupleType tup) -> std::optional<Type> {
    SmallVector<Type,4> newElems;
    newElems.reserve(tup.size());
    for (Type elem : tup.getTypes()) {
      SmallVector<Type,2> converted;
      if (failed(converter.convertType(elem, converted)))
        return std::nullopt;

      // append all converted pieces (0 to N). 0 means the element was erased
      newElems.append(converted);
    }
    return TupleType::get(tup.getContext(), newElems);
  });

  patterns.add<
    EraseClaimsFromGetOp,
    EraseClaimsFromMakeOp
  >(converter, patterns.getContext());
}

} // end mlir::tuple
