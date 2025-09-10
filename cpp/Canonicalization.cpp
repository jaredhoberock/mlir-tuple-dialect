#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <Instantiation.hpp>

namespace mlir::tuple {

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

void populateTupleCanonicalizationPatterns(RewritePatternSet& patterns) {
  patterns.add<
    AppendOpCanonicalization,
    CmpOpCanonicalization,
    GetOpCanonicalization
  >(patterns.getContext());
}

} // end mlir::tuple
