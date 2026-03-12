// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Canonicalization.hpp"
#include "Monomorphization.hpp"
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Transforms/DialectConversion.h>
#include <Instantiation.hpp>
#include <Trait.hpp>

namespace mlir::tuple {

/// This file defines several distinct pattern sets, each responsible for a
/// different phase of lowering. They must remain clearly separated to preserve
/// invariants about polymorphism, trait integration, and tuple structure.
///
/// 1. populateConvertTupleToTraitPatterns
///    -----------------------------------
///    Introduces tuple→trait integration IR that *cannot* appear during
///    instantiation. This includes generating helper traits and synthesizing
///    per-element trait claims. Runs before monomorphization.
///
/// 2. populateTupleElaborationPatterns
///    -------------------------------
///    Elaborates higher-level tuple operations into a uniform set of primitive
///    tuple constructs. These patterns stay entirely inside the tuple dialect.
///    They do not instantiate polymorphism and do not introduce trait-level IR.
///    They simply normalize tuple IR into a structurally explicit form.
///
/// 3. populateInstantiateMonomorphsPatterns
///    -------------------------------------
///    Specializes polymorphic tuple operations once shapes and substitutions are
///    known. This phase unrolls higher-order tuple ops, performs body
///    instantiation, and produces fully monomorphic, first-order tuple IR.
///    Includes in-dialect elaboration patterns because instantiation may reveal
///    more tuple structure to rewrite.
///
/// 4. populateErasePolymorphsPatterns
///    --------------------------------
///    Cooperates with a TypeConverter to remove `!trait.claim` types from the IR
///    after trait reasoning is complete. Adjusts tuple IR (indices, make ops,
///    etc.) to account for elements that are erased or expanded.


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

    // XXX TODO we shouldn't assume that the trait to be mapped has two type parameters

    // create:
    //
    // !S = trait.poly<unique>
    // !O = trait.poly<unique>
    // !C = trait.poly<unique>
    // trait.trait @tuple.Map<mapped-trait-name>[!S,!O,!C] attributes {
    //   tuple.impl_generator = "map",
    //   tuple.mapped_trait = @<mapped-trait-name>
    // } {
    //   func.func private @claims() -> !C
    // }

    Type S = trait::PolyType::getUnique(ctx);
    Type O = trait::PolyType::getUnique(ctx);
    Type C = trait::PolyType::getUnique(ctx);
    
    auto trait = rewriter.create<trait::TraitOp>(
      loc,
      StringAttr::get(ctx, name),
      /*typeParams=*/ArrayRef{S, O, C},
      /*requirements=*/trait::TraitApplicationArrayAttr::get(ctx, {})
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


/// Register patterns that introduce trait-specific stuff that *cannot*
/// be introduced during instantiation/monomorphization
void populateConvertTupleToTraitPatterns(RewritePatternSet& patterns) {
  // introduce the @tuple.MapPartialEq and @tuple.MapPartialOrd traits
  // that drive tuple-level implementations of these traits
  patterns.add<IntroduceMapperTrait>(patterns.getContext(), "PartialEq");
  patterns.add<IntroduceMapperTrait>(patterns.getContext(), "PartialOrd");

  // these patterns introduce trait.allege ops, which cannot happen
  // during monomorphization
  patterns.add<CmpOpMonoSynthesizeClaims>(patterns.getContext());
}


//===----------------------------------------------------------------------===//
// populateTupleElaborationPatterns
//===----------------------------------------------------------------------===//

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
    Value init = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));

    // tuple.foldl expects: result type, init, inputs...
    // our single input is the tuple being tested by tuple.all
    auto fold = rewriter.create<FoldlOp>(
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
      Value both = rewriter.create<arith::AndIOp>(loc, newBody->getArgument(0), pred);

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
      elems.push_back(rewriter.create<GetOp>(loc, op.getTuple(), i));
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
      elems.push_back(rewriter.create<GetOp>(
        loc,
        elemTy,
        op.getLhs(),
        rewriter.getIndexAttr(i)
      ));
    }

    // emit gets for rhs elements
    for (unsigned i = 0, e = rhsTT.size(); i < e; ++i) {
      Type elemTy = rhsTT.getType(i);
      elems.push_back(rewriter.create<GetOp>(
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

      Type Li = trait::PolyType::getUnique(ctx);
      Type Ri = trait::PolyType::getUnique(ctx);
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
      elems.push_back(rewriter.create<GetOp>(loc, op.getInput(), i));
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
    auto first = rewriter.create<MakeOp>(loc, ValueRange{op.getInit()});

    auto fold = rewriter.create<FoldlOp>(
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
      Value acc = rewriter.create<LastOp>(loc, prev);

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
      Value next = rewriter.create<AppendOp>(loc, prev, yielded);

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
    auto mapOp = rewriter.create<MapOp>(loc, *mapResultTy, ValueRange{op.getInput()});
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


static void populateTupleElaborationPatterns(RewritePatternSet& patterns) {
  // all of these patterns introduce rewrites that stay within the tuple dialect
  patterns.add<
    AllOpLowering,
    AppendOpLowering,
    CatOpLowering,
    CmpOpPartialEqLowering,
    CmpOpPartialOrdLowering,
    DropLastOpLowering,
    ExclusiveScanOpLowering,
    FlatMapOpLowering,
    LastOpLowering
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


struct FlattenOpInstantiation : public OpRewritePattern<FlattenOp> {
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
      Value inner = rewriter.create<GetOp>(loc, input, outerIdx);

      // extract each element of the inner tuple and append to flatElems
      for (unsigned j = 0; j < innerTT.size(); ++j) {
        flatElems.push_back(rewriter.create<GetOp>(loc, inner, j));
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
FailureOr<Value> instantiateElementalTupleOpIteration(
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

    SmallVector<Value> resultElems;
    resultElems.reserve(arity);

    for (unsigned i = 0; i < arity; ++i) {
      // collect the ith element from each input tuple to pass to the block
      SmallVector<Value> args;
      args.reserve(op.getInputs().size());
      for (Value tup : op.getInputs()) {
        args.push_back(rewriter.create<GetOp>(loc, tup, i));
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


void populateInstantiateMonomorphsPatterns(RewritePatternSet& patterns) {
  // instantiating monomorphs may generate new tuple ops,
  // so include all the in-dialect elaboration patterns
  populateTupleElaborationPatterns(patterns);

  // tuple.flatten, tuple.foldl and tuple.map
  // lowerings instantiate / specialize tuple structure.
  patterns.add<
    FlattenOpInstantiation,
    FoldlOpInstantiation,
    MapOpInstantiation
  >(patterns.getContext());
}


//===----------------------------------------------------------------------===//
// populateErasePolymorphsPatterns
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

/// Register patterns that cooperate with a TypeConverter to *erase*
/// `!trait.claim` types from tuples and the IR.
///
/// The provided TypeConverter is responsible for mapping:
///   - `!trait.claim<...>` -> (0 pieces)
///   - other types -> 1+ pieces
///
/// This pattern set:
///   - updates `tuple.get` indices to account for elements that were
///     expanded or erased under the type conversion,
///   - rebuilds `tuple.make` with the converted operand lists and result
///     types, dropping operands whose types erased to nothing.
///
/// After this phase, there should be no `!trait.claim` types remaining in
/// tuple element types or in SSA value types.
void populateErasePolymorphsPatterns(TypeConverter& converter, RewritePatternSet& patterns) {
  // teach the TypeConverter how to rewrite TupleType by recursively applying
  // the element conversion (including erasing elements that convert to 0 pieces)
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

  // rewrite tuple ops to respect the converted types:
  // - GetOp: adjust indices and optionally erase tuple.get ops whose result type
  //   erased to nothing
  // - MakeOp: rebuild tuples from the converted operands and result type,
  //   flattening multi-piece operands and dropping erased ones
  patterns.add<
    EraseClaimsFromGetOp,
    EraseClaimsFromMakeOp
  >(converter, patterns.getContext());
}

} // end mlir::tuple
