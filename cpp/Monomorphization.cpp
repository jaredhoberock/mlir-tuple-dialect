// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Canonicalization.hpp"
#include "Elaboration.hpp"
#include "Monomorphization.hpp"
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Transforms/DialectConversion.h>
#include <Instantiation.hpp>
#include <Trait.hpp>

namespace mlir::tuple {

/// This file defines two pattern sets that integrate the tuple dialect with
/// trait-level monomorphization. The pure in-dialect elaboration of
/// higher-level tuple ops into the `tuple.make` / `tuple.get` core lives in
/// Elaboration.cpp and is reused from here via
/// `populateTupleElaborationPatterns`.
///
/// 1. populateConvertTupleToTraitPatterns
///    -----------------------------------
///    Introduces tuple→trait integration IR that *cannot* appear during
///    instantiation. This includes generating helper traits and synthesizing
///    per-element trait claims. Runs before monomorphization.
///
/// 2. populateInstantiateMonomorphsPatterns
///    -------------------------------------
///    Specializes polymorphic tuple operations once shapes and substitutions
///    are known. Higher-order region-bearing ops are unrolled per element
///    via in-dialect elaboration (Elaboration.cpp). This set is contributed
///    to the trait dialect's `monomorphize-trait` pipeline through the
///    MonomorphizationInterface.
///
/// 3. populateErasePolymorphsPatterns
///    --------------------------------
///    Cooperates with a TypeConverter to remove `!trait.claim` types from the
///    IR after trait reasoning is complete. Adjusts tuple IR (indices, make
///    ops, etc.) to account for elements that are erased or expanded.


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

    auto trait = trait::TraitOp::create(rewriter,
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

      auto claimsFn = func::FuncOp::create(rewriter,
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
      Value ci = trait::AllegeOp::create(rewriter, loc, app);
      claimElems.push_back(ci);
    }

    // %claims = tuple.make (c1..ck)
    Value claimsTuple = MakeOp::create(rewriter, loc,claimElems);

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
// populateInstantiateMonomorphsPatterns
//===----------------------------------------------------------------------===//

void populateInstantiateMonomorphsPatterns(RewritePatternSet& patterns) {
  // The in-dialect elaboration of higher-level tuple ops into the
  // tuple.make / tuple.get core is the same work needed for
  // monomorphization, so reuse it directly.
  populateTupleElaborationPatterns(patterns);
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
