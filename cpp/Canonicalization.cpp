// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Canonicalization.hpp"
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <Instantiation.hpp>

namespace mlir::tuple {

struct AllOpCanonicalization : public OpRewritePattern<AllOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AllOp allOp,
                                PatternRewriter& rewriter) const override {
    // if the arity is known to be zero, all -> true
    if (auto arity = allOp.getArity(); arity && *arity == 0) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(allOp, rewriter.getBoolAttr(true));
      return success();
    }

    return failure();
  }
};

struct AppendOpCanonicalization : public OpRewritePattern<AppendOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AppendOp appendOp,
                                PatternRewriter& rewriter) const override {
    // check if the input tuple came from tuple.make
    auto makeOp = appendOp.getTuple().getDefiningOp<MakeOp>();
    if (!makeOp)
      return failure();

    // replace with tuple.make
    SmallVector<Value> elements(makeOp.getElements());
    elements.push_back(appendOp.getElement());

    rewriter.replaceOpWithNewOp<MakeOp>(
      appendOp,
      elements
    );

    return success();
  }
};

struct CatOpCanonicalization : public OpRewritePattern<CatOp> {
  using OpRewritePattern::OpRewritePattern;

  static bool isEmptyTupleValue(Value v) {
    if (auto tt = dyn_cast<TupleType>(v.getType()))
      if (tt.size() == 0) return true;
    return false;
  }

  LogicalResult matchAndRewrite(CatOp op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    // check for empty tuples
    // cat (), rhs -> rhs
    if (isEmptyTupleValue(lhs)) {
      rewriter.replaceOp(op, rhs);
      return success();
    }

    // cat lhs, () -> lhs
    if (isEmptyTupleValue(rhs)) {
      rewriter.replaceOp(op, lhs);
      return success();
    }

    // fuse adjacent makes
    // cat make(xs...), make(ys...) -> make(xs..., ys...)
    if (auto lhsMake = lhs.getDefiningOp<tuple::MakeOp>()) {
      if (auto rhsMake = rhs.getDefiningOp<tuple::MakeOp>()) {
        SmallVector<Value> elems;
        elems.reserve(lhsMake->getNumOperands() + rhsMake->getNumOperands());
        elems.append(lhsMake->operand_begin(), lhsMake->operand_end());
        elems.append(rhsMake->operand_begin(), rhsMake->operand_end());

        rewriter.replaceOpWithNewOp<tuple::MakeOp>(op, elems);
        return success();
      }
    }

    // reassociate: cat(cat(a,b), c) -> cat(a, cat(b,c))
    if (auto innerCat = lhs.getDefiningOp<CatOp>()) {
      Value a = innerCat.getLhs();
      Value b = innerCat.getRhs();

      auto bc = CatOp::create(rewriter, op.getLoc(), b, rhs);
      rewriter.replaceOpWithNewOp<CatOp>(op, a, bc.getResult());
      return success();
    }

    return failure();
  }
};

struct CmpOpCanonicalization : public OpRewritePattern<CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter& rewriter) const override {
    // check if lhs & rhs are both empty tuples
    auto arity = op.getArity();
    if (!arity)
      return rewriter.notifyMatchFailure(op, "unknown arity");

    // check if the arity is 0
    if (*arity != 0)
      return rewriter.notifyMatchFailure(op, "arity is not zero");

    // empty tuples fold to a constant depending on the predicate:
    bool value = false;
    switch (op.getPredicate()) {
      case CmpPredicate::eq:
      case CmpPredicate::le:
      case CmpPredicate::ge:
        value = true;
        break;
      case CmpPredicate::ne:
      case CmpPredicate::lt:
      case CmpPredicate::gt:
        value = false;
        break;
    }

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
      op,
      rewriter.getBoolAttr(value)
    );

    return success();
  }
};

struct GetOpCanonicalization : public OpRewritePattern<GetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GetOp op,
                                PatternRewriter& rewriter) const override {
    // check if the input tuple came from tuple.make
    auto makeOp = op.getTuple().getDefiningOp<MakeOp>();
    if (!makeOp)
      return failure();

    // get the ith element
    int64_t i = op.getIndex().getSExtValue();
    Value element = makeOp.getElements()[i];

    // replace with the ith element
    rewriter.replaceOp(op, element);
    return success();
  }
};

struct MakeOpCanonicalization : public OpRewritePattern<MakeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeOp op,
                                PatternRewriter& rewriter) const override {
    auto resultTy = dyn_cast<TupleType>(op.getResult().getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "result is not a TupleType");

    unsigned n = op.getNumOperands();
    if (n == 0)
      return rewriter.notifyMatchFailure(op, "no operands");

    // match a tuple.make built from contiguous tuple.get from the same original tuple
    Value origin;
    SmallVector<unsigned, 4> indices;
    indices.reserve(n);

    for (unsigned i = 0; i < n; ++i) {
      auto *def = op.getOperand(i).getDefiningOp();
      auto get = dyn_cast_or_null<GetOp>(def);
      if (!get)
        return rewriter.notifyMatchFailure(op, "operand is not a tuple.get");

      if (!origin)
        origin = get.getTuple();
      else if (origin != get.getTuple())
        return rewriter.notifyMatchFailure(op, "operands come from different origins");

      auto idx = get.getIndex().getSExtValue();
      indices.push_back(idx);
    }

    // require contiguous 0..n-1
    for (unsigned i = 0; i < n; ++i) {
      if (indices[i] != i)
        return rewriter.notifyMatchFailure(op, "indices are not 0..n-1 in order");
    }

    // original type must match result type.
    if (origin.getType() != resultTy)
      return rewriter.notifyMatchFailure(op, "original type does not match result type");

    rewriter.replaceOp(op, origin);
    return success();
  }
};

void populateTupleCanonicalizationPatterns(RewritePatternSet& patterns) {
  patterns.add<
    AllOpCanonicalization,
    AppendOpCanonicalization,
    CatOpCanonicalization,
    CmpOpCanonicalization,
    GetOpCanonicalization,
    MakeOpCanonicalization
  >(patterns.getContext());
}

void TupleCanonicalizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateTupleCanonicalizationPatterns(patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

} // end mlir::tuple
