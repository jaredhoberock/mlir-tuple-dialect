#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir-trait-dialect/cpp/Instantiation.hpp>

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

struct FoldlOpCanonicalization : public OpRewritePattern<FoldlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FoldlOp op,
                                PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();

    Region &body = op.getBody();
    Block &entry = body.front();

    Value previousResult = op.getInit();
    for (unsigned int i = 0; i < op.getArity(); ++i) {
      // get the ith element from the input tuple
      Value element = rewriter.create<GetOp>(loc, op.getTuple(), i);

      // build the type substitution for this iteration
      DenseMap<Type,Type> substitution = op.buildSubstitutionForIteration(i, previousResult.getType());

      // instantiate the body into a temporary Region
      Region bodyInstance;
      trait::instantiatePolymorphicRegion(rewriter, body, bodyInstance, substitution);

      // get the block to inline
      Block* blockToInline = &bodyInstance.front();

      // before inlining, find the block's YieldOp
      YieldOp yieldOp = cast<YieldOp>(blockToInline->getTerminator());

      // inline the block, replacing block arguments with the previous result and tuple element
      rewriter.inlineBlockBefore(blockToInline, op, {previousResult, element});

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

struct MapOpCanonicalization : public OpRewritePattern<MapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp mapOp,
                                PatternRewriter& rewriter) const override {
    Location loc = mapOp.getLoc();
    unsigned int arity = mapOp.getArity();

    Region &body = mapOp.getBody();
    Block &entry = body.front();

    SmallVector<Value> results;
    for (unsigned int i = 0; i < arity; ++i) {
      // get the ith element from each input tuple
      SmallVector<Value> ithElements;
      for (Value tuple : mapOp.getInputs()) {
        ithElements.push_back(rewriter.create<GetOp>(loc, tuple, i));
      }

      // build the type substitution for this iteration
      DenseMap<Type,Type> substitution = mapOp.buildSubstitutionForIteration(i);

      // instantiate the body into a temporary Region
      Region bodyInstance;
      trait::instantiatePolymorphicRegion(rewriter, body, bodyInstance, substitution);

      // get the block to inline
      Block* blockToInline = &bodyInstance.front();

      // before inlining, find the block's YieldOp
      YieldOp yieldOp = cast<YieldOp>(blockToInline->getTerminator());

      // inline the block, replacing block arguments with the ith element of each tuple
      rewriter.inlineBlockBefore(blockToInline, mapOp, ithElements);

      // now that the yield op has been inlined, grab its operand
      results.push_back(yieldOp->getOperand(0));

      // erase the yield
      rewriter.eraseOp(yieldOp);
    }

    // replace the map op with the assembled result tuple
    rewriter.replaceOpWithNewOp<MakeOp>(mapOp, results);

    return success();
  }
};

void populateTupleCanonicalizationPatterns(RewritePatternSet& patterns) {
  patterns.add<
    AppendOpCanonicalization,
    FoldlOpCanonicalization,
    MapOpCanonicalization
  >(patterns.getContext());
}

} // end mlir::tuple
