// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Elaboration.hpp"
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <Instantiation.hpp>
#include <Trait.hpp>

namespace mlir::tuple {

// rewrites tuple.all into a tuple.foldl op
struct AllOpLowering : OpRewritePattern<AllOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    Type i1Ty = rewriter.getI1Type();

    // if the arity is known to be zero, `all` is vacuously true
    if (auto arity = op.getArity(); arity && *arity == 0) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, rewriter.getBoolAttr(true));
      return success();
    }

    // build init = true and create tuple.foldl over the input tuple
    Value init = arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));

    // tuple.foldl expects: result type, init, inputs...
    // our single input is the tuple being tested by tuple.all
    auto fold = FoldlOp::create(rewriter,
      loc,
      /*resultTy=*/i1Ty,
      init,
      /*inputs=*/ValueRange{op.getInput()}
    );

    // we will splice the all-body block into the foldl region, then:
    // - insert an accumulator arg (i1) at position 0
    // - AND the old yielded predicate with the accumulator
    // - yield the conjunction
    {
      PatternRewriter::InsertionGuard guard(rewriter);

      // new block inside fold with (%acc, %elem)
      Block *newBody = rewriter.createBlock(&fold.getBody());
      Type elemFormalTy = op.getBody().front().getArgument(0).getType();
      newBody->addArguments({i1Ty, elemFormalTy}, {loc, loc});

      // move original body ops into the new block, remapping %oldElem -> %elem
      Block &oldBody = op.getBody().front();
      rewriter.mergeBlocks(&oldBody, newBody, /*argValues=*/ValueRange{newBody->getArgument(1)});

      // grab the (now moved) yield and AND its operand with %acc
      auto oldYield = fold.bodyYield();
      Value pred = oldYield.getOperand();

      rewriter.setInsertionPoint(oldYield);
      Value both = arith::AndIOp::create(rewriter, loc, newBody->getArgument(0), pred);

      // Replace yield with a fresh yield of %both
      rewriter.replaceOpWithNewOp<tuple::YieldOp>(oldYield, both);
    }

    rewriter.replaceOp(op, fold.getResult());
    return success();
  }
};

// rewrites tuple.append into tuple.make once the arity of the input tuple is known
struct AppendOpLowering : OpRewritePattern<AppendOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AppendOp op,
                                PatternRewriter &rewriter) const override {
    // we lower only when the input is TupleType
    auto inputTupleTy = dyn_cast_or_null<TupleType>(op.getTuple().getType());
    if (!inputTupleTy) {
      return rewriter.notifyMatchFailure(op, "unsupported input tuple type");
    }

    auto loc = op.getLoc();
    unsigned arity = inputTupleTy.size();

    SmallVector<Value> elems;
    elems.reserve(arity + 1);

    for (unsigned i = 0; i < arity; ++i) {
      elems.push_back(GetOp::create(rewriter, loc, op.getTuple(), i));
    }
    elems.push_back(op.getElement());

    rewriter.replaceOpWithNewOp<MakeOp>(op, elems);
    return success();
  }
};

// rewrites tuple.cat into tuple.make once the arity of inputs are known
struct CatOpLowering : OpRewritePattern<CatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CatOp op,
                                PatternRewriter& rewriter) const override {
    // we only lower when both operands are TupleType
    auto lhsTT = dyn_cast<TupleType>(op.getLhs().getType());
    auto rhsTT = dyn_cast<TupleType>(op.getRhs().getType());
    if (!lhsTT || !rhsTT)
      return rewriter.notifyMatchFailure(op, "operands are not TupleType");

    Location loc = op.getLoc();
    SmallVector<Value, 8> elems;
    elems.reserve(lhsTT.size() + rhsTT.size());

    // emit gets for lhs elements
    for (unsigned i = 0, e = lhsTT.size(); i < e; ++i) {
      Type elemTy = lhsTT.getType(i);
      elems.push_back(GetOp::create(rewriter,
        loc,
        elemTy,
        op.getLhs(),
        rewriter.getIndexAttr(i)
      ));
    }

    // emit gets for rhs elements
    for (unsigned i = 0, e = rhsTT.size(); i < e; ++i) {
      Type elemTy = rhsTT.getType(i);
      elems.push_back(GetOp::create(rewriter,
        loc,
        elemTy,
        op.getRhs(),
        rewriter.getIndexAttr(i)
      ));
    }

    // rebuild as a single tuple.make
    rewriter.replaceOpWithNewOp<MakeOp>(op, elems);
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
    Value init = arith::ConstantOp::create(rewriter,
      loc,
      rewriter.getBoolAttr(method == "eq")
    );

    SmallVector<Value,3> inputs{op.getLhs(), op.getRhs(), claims};
    auto fold = FoldlOp::create(rewriter, loc, i1Ty, init, inputs);

    // build the body:
    //
    // ^bb0(%acc: i1, %li: !Li, %ri: !Ri, %ci: !trait.claim<@PartialEq[!Li,!Ri]>):
    //   %call_res = trait.method.call %ci @PartialEq[!Li,!Ri]::method(%li, %ri) : ...
    //   %resi = AND or OR with %acc
    //   yield %resi : i1
    {
      Type Li = trait::PolyType::getUnique(ctx);
      Type Ri = trait::PolyType::getUnique(ctx);
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
      Value callRes = trait::MethodCallOp::create(rewriter,
        loc, i1Ty,
        op.getTraitName(),
        method,
        ci,
        ValueRange{li,ri}
      ).getResult(0);

      // fold: eq -> AND; ne -> OR
      Value resi = (method == "eq")
        ? (Value)arith::AndIOp::create(rewriter, loc, acc, callRes)
        : (Value)arith::OrIOp::create(rewriter, loc, acc, callRes);

      YieldOp::create(rewriter, loc, resi);
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
    Value cFalse = arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
    Value cTrue  = arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
    Value init   = MakeOp::create(rewriter, loc, accTy, ValueRange{cFalse, cTrue});

    // build foldl over (lhs, rhs, claims)
    SmallVector<Value,3> inputs{op.getLhs(), op.getRhs(), claims};
    auto fold = FoldlOp::create(rewriter, loc, accTy, init, inputs);

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

      Type Li = trait::PolyType::getUnique(ctx);
      Type Ri = trait::PolyType::getUnique(ctx);
      Type Ci = trait::ClaimType::get(ctx, op.getTraitRefAttr(), {Li, Ri});

      Block *body = rewriter.createBlock(&fold.getBody());
      body->addArguments({accTy, Li, Ri, Ci}, {loc, loc, loc, loc});

      Value acc = body->getArgument(0);
      Value li = body->getArgument(1);
      Value ri = body->getArgument(2);
      Value ci = body->getArgument(3);

      Value res   = GetOp::create(rewriter, loc, i1, acc, rewriter.getIndexAttr(0));
      Value allEq = GetOp::create(rewriter, loc, i1, acc, rewriter.getIndexAttr(1));

      Value lt_i = trait::MethodCallOp::create(rewriter,
        loc, i1, "PartialOrd", "lt", ci, ValueRange{li, ri}).getResult(0);
      Value gt_i = trait::MethodCallOp::create(rewriter,
        loc, i1, "PartialOrd", "gt", ci, ValueRange{li, ri}).getResult(0);

      Value either = arith::OrIOp::create(rewriter, loc, lt_i, gt_i);
      Value eq_i   = arith::XOrIOp::create(rewriter, loc, either, cTrue);

      Value strict = isLtLe ? lt_i : gt_i;

      Value contrib     = arith::AndIOp::create(rewriter, loc, allEq, strict);
      Value res_next    = arith::OrIOp::create(rewriter, loc, res, contrib);
      Value all_eq_next = arith::AndIOp::create(rewriter, loc, allEq, eq_i);

      Value acc_next = MakeOp::create(rewriter, loc, accTy, ValueRange{res_next, all_eq_next});
      YieldOp::create(rewriter, loc, acc_next);
    }

    // finalize:
    // lt/gt -> finalRes = res
    // le/ge -> finalRes = res || allEq
    Value finalRes = GetOp::create(rewriter, loc, i1, fold.getResult(), rewriter.getIndexAttr(0));
    if (!isStrict) {
      Value finalAllEq = GetOp::create(rewriter, loc, i1, fold.getResult(), rewriter.getIndexAttr(1));
      finalRes = arith::OrIOp::create(rewriter, loc, finalRes, finalAllEq);
    }

    rewriter.replaceOp(op, finalRes);
    return success();
  }
};

struct DropLastOpLowering : OpRewritePattern<DropLastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DropLastOp op,
                                PatternRewriter &rewriter) const override {
    auto inputTupleTy = dyn_cast<TupleType>(op.getInput().getType());
    if (!inputTupleTy)
      return rewriter.notifyMatchFailure(op, "input is not TupleType");

    Location loc = op.getLoc();
    unsigned arity = inputTupleTy.size();
    if (arity < 1)
      return rewriter.notifyMatchFailure(op, "input tuple arity is < 1");

    SmallVector<Value> elems;
    elems.reserve(arity - 1);

    for (unsigned i = 0; i < arity - 1; ++i) {
      elems.push_back(GetOp::create(rewriter, loc, op.getInput(), i));
    }

    rewriter.replaceOpWithNewOp<MakeOp>(op, elems);
    return success();
  }
};

struct ExclusiveScanOpLowering : public OpRewritePattern<ExclusiveScanOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExclusiveScanOp op,
                                PatternRewriter& rewriter) const override {
    // replace tuple.exclusive_scan with tuple.foldl:

    // %res = tuple.exclusive_scan %input, %init : !T, !I -> !R {
    // ^bb0(%acc: !A, %e: !E):
    //   %yielded = ... : !Y
    //   yield %yielded : !Y
    // }
    //
    // =>
    //
    // !P = !tuple.poly<unique>
    // !N = !tuple.poly<unique> // this type will be inferred
    //
    // %first = tuple.make (%init : !I) -> tuple<!I>
    // %res = tuple.foldl %first, %input : tuple<!I>, !T -> !R {
    // ^bb0(%prev: !P, %e: !E):
    //   %acc = tuple.last %prev : !P -> !A
    //   %yielded = ... : !Y
    //   %next = tuple.append %prev, %yielded : !P, !Y -> !N
    //   yield %next : !N
    // }

    auto loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    // initial value of the prefix state: (init)
    auto first = MakeOp::create(rewriter, loc, ValueRange{op.getInit()});

    auto fold = FoldlOp::create(rewriter,
        loc,
        /*resultTy=*/op.getResult().getType(),
        /*init=*/first,
        /*inputs=*/op.getInput()
    );
    {
      PatternRewriter::InsertionGuard guard(rewriter);

      // original exclusive scan body: ^bb0(%acc: !A, %e: !E)
      Block &oldBody = op.getBody().front();
      Type elemTy = oldBody.getArgument(1).getType();

      // polymorphic prefix state type:
      // !P = !tuple.poly<unique>
      Type stateTy = PolyType::getUnique(ctx);

      // new fold body:
      // ^bb0(%prev: !P, %e: !E)
      Block *newBody = rewriter.createBlock(&fold.getBody());
      newBody->addArguments({stateTy, elemTy}, {loc, loc});

      Value prev = newBody->getArgument(0);
      Value elem = newBody->getArgument(1);

      // %acc = tuple.last %prev : !P -> !A
      rewriter.setInsertionPointToStart(newBody);
      Value acc = LastOp::create(rewriter, loc, prev);

      // inline the original body, remapping:
      // old %acc -> %acc
      // old %e   -> %elem
      rewriter.mergeBlocks(
        &oldBody,
        newBody,
        ValueRange{acc, elem}
      );

      // after merge: we have the old yield inside the new block
      auto oldYield = cast<YieldOp>(fold.bodyYield());
      Value yielded = oldYield.getOperand();

      // %next = tuple.append %prev, %yielded : !P, !Y -> !N
      rewriter.setInsertionPoint(oldYield);
      Value next = AppendOp::create(rewriter, loc, prev, yielded);

      // yield %next : !N
      rewriter.replaceOpWithNewOp<YieldOp>(oldYield, next);
    }

    // replace the original op with fold result
    rewriter.replaceOp(op, fold.getResult());

    return success();
  }
};

struct FlatMapOpLowering : public OpRewritePattern<FlatMapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FlatMapOp op,
                                PatternRewriter& rewriter) const override {
    auto mapResultTy = op.inferIntermediateMapType();
    if (failed(mapResultTy))
      return rewriter.notifyMatchFailure(op, "inferIntermediateMapType failed");

    Location loc = op.getLoc();

    // create tuple.map with the same body as the tuple.flat_map
    auto mapOp = MapOp::create(rewriter, loc, *mapResultTy, ValueRange{op.getInput()});
    {
      // inline the flat_map's body into the new map op
      Region &oldBody = op.getBody();
      Region &newBody = mapOp.getBody();
      rewriter.inlineRegionBefore(oldBody, newBody, newBody.end());
    }

    // replace tuple.flat_map with tuple.flatten of the map result
    Type flatResTy = op.getResult().getType();
    rewriter.replaceOpWithNewOp<FlattenOp>(op, flatResTy, mapOp.getResult());
    return success();
  }
};

struct LastOpLowering : OpRewritePattern<LastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LastOp op,
                                PatternRewriter &rewriter) const override {
    auto inputTupleTy = dyn_cast<TupleType>(op.getInput().getType());
    if (!inputTupleTy)
      return rewriter.notifyMatchFailure(op, "input is not TupleType");

    if (inputTupleTy.size() == 0)
      return rewriter.notifyMatchFailure(op, "input is empty tuple");

    rewriter.replaceOpWithNewOp<GetOp>(op, op.getInput(), inputTupleTy.size() - 1);
    return success();
  }
};

struct FoldlOpLowering : public OpRewritePattern<FoldlOp> {
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
        args.push_back(GetOp::create(rewriter, loc, tuple, i));

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


struct FlattenOpLowering : public OpRewritePattern<FlattenOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FlattenOp op,
                                PatternRewriter& rewriter) const override {
    // only lower once both input and result and concrete TupleType
    auto inTT = dyn_cast<TupleType>(op.getInput().getType());
    auto outTT = dyn_cast<TupleType>(op.getResult().getType());
    if (!inTT || !outTT)
      return rewriter.notifyMatchFailure(
          op, "input/result are not concrete TupleType");

    Location loc = op.getLoc();
    Value input = op.getInput();

    SmallVector<Value, 8> flatElems;
    flatElems.reserve(outTT.size());

    // for each outer element, which must itself be a concrete TupleType,
    // grab the inner tuple and then each of its elements
    for (auto [outerIdx, outerElemTy] : llvm::enumerate(inTT.getTypes())) {
      auto innerTT = dyn_cast<TupleType>(outerElemTy);
      if (!innerTT)
        return rewriter.notifyMatchFailure(
            op, "element of input tuple is not a concrete TupleType");

      // %inner = tuple.get %input, outerIdx
      Value inner = GetOp::create(rewriter, loc, input, outerIdx);

      // extract each element of the inner tuple and append to flatElems
      for (unsigned j = 0; j < innerTT.size(); ++j) {
        flatElems.push_back(GetOp::create(rewriter, loc, inner, j));
      }
    }

    // sanity: arity should match what the verifier computed
    if (flatElems.size() != outTT.size())
      return rewriter.notifyMatchFailure(
          op, "flattened arity does not match result type");

    rewriter.replaceOpWithNewOp<MakeOp>(op, outTT, flatElems);
    return success();
  }
};


// instantiate one *elemental* iteration of a higher-order tuple op (map or flat_map).
// - asks the op for the substitution for iteration `i`
// - instantiates the body into a temporary region
// - inlines the instantiated block before `op`, binding `args` to the block args
// - returns the Value yielded by that iteration
template<typename OpT>
static FailureOr<Value> instantiateElementalTupleOpIteration(
    PatternRewriter &rewriter,
    OpT op,
    unsigned i,
    ArrayRef<Value> args) {

  // guard the insertion point
  PatternRewriter::InsertionGuard guard(rewriter);

  // get the substitution for iteration i
  auto subst = op.buildSubstitutionForIteration(i);
  if (failed(subst)) return failure();

  Region &body = op.getBody();

  // instantiate body into temporary region
  Region bodyInstance;
  trait::instantiatePolymorphicRegion(rewriter, body, bodyInstance, *subst);

  Block *block = &bodyInstance.front();
  auto yieldOp = cast<YieldOp>(block->getTerminator());

  // inline instantiated block, binding args
  rewriter.inlineBlockBefore(block, op, args);

  // grab yielded value
  Value yielded = yieldOp.getOperand();

  // erase old yield op
  rewriter.eraseOp(yieldOp);

  return yielded;
}

struct MapOpLowering : public OpRewritePattern<MapOp> {
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

    SmallVector<Value> resultElems;
    resultElems.reserve(arity);

    for (unsigned i = 0; i < arity; ++i) {
      // collect the ith element from each input tuple to pass to the block
      SmallVector<Value> args;
      args.reserve(op.getInputs().size());
      for (Value tup : op.getInputs()) {
        args.push_back(GetOp::create(rewriter, loc, tup, i));
      }

      // instantiate and inline one iteration of the body
      auto yielded = instantiateElementalTupleOpIteration(rewriter, op, i, args);
      if (failed(yielded))
        return rewriter.notifyMatchFailure(op, "instantiateElementalTupleOpIteration failed");

      // the yielded value is the ith element of the result tuple
      resultElems.push_back(*yielded);
    }

    // replace the map op with the assembled result tuple
    rewriter.replaceOpWithNewOp<MakeOp>(op, resultElems);
    return success();
  }
};


void populateTupleElaborationPatterns(RewritePatternSet& patterns) {
  patterns.add<
    AllOpLowering,
    AppendOpLowering,
    CatOpLowering,
    CmpOpPartialEqLowering,
    CmpOpPartialOrdLowering,
    DropLastOpLowering,
    ExclusiveScanOpLowering,
    FlatMapOpLowering,
    FlattenOpLowering,
    FoldlOpLowering,
    LastOpLowering,
    MapOpLowering
  >(patterns.getContext());
}


//===----------------------------------------------------------------------===//
// TupleElaboratePass
//===----------------------------------------------------------------------===//

void TupleElaboratePass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  populateTupleElaborationPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // end mlir::tuple
