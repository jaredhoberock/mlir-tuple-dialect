#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir-trait-dialect/cpp/Instantiation.hpp>

namespace mlir::tuple {

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
    rewriter.replaceOpWithNewOp<ConstantOp>(mapOp, results);

    return success();
  }
};

void TupleDialect::getCanonicalizationPatterns(RewritePatternSet& patterns) const {
  patterns.add<MapOpCanonicalization>(patterns.getContext());
}

} // end mlir::tuple
