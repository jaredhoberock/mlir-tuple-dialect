#include "ConvertToTrait.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace mlir::tuple {

static trait::ImplOp getOrCreatePartialEqImpl(OpBuilder& builder, trait::TraitOp partialEqOp) {
  FlatSymbolRefAttr partialEqRefAttr =
    FlatSymbolRefAttr::get(builder.getContext(), "PartialEq");

  // construct the receiver type
  // !T = !tuple.any<[@PartialEq]>
  Type receiverTy =
    AnyTupleType::get(builder.getContext(), {partialEqRefAttr});

  // check if the impl already exists for this receiver
  if (auto existing = partialEqOp.getImpl(receiverTy)) return existing;

  // the impl doesn't exist yet; create it

  // create this op:
  //
  // !T = !tuple.any<[@PartialEq]>
  // trait.impl @PartialEq for !T {
  //   func.func private @eq(%self: !T, %other: !T) -> i1 {
  //     %res = tuple.cmp eq, %self, %other : !T
  //     return %res : i1
  //   }
  // }

  PatternRewriter::InsertionGuard guard(builder);

  MLIRContext* ctx = builder.getContext();
  Location loc = partialEqOp.getLoc();

  // create trait.impl @PartialEq for !T {}
  builder.setInsertionPointAfter(partialEqOp);
  auto implOp = builder.create<trait::ImplOp>(
      loc,
      partialEqRefAttr,
      TypeAttr::get(receiverTy)
  );

  // get the pre-created entry block
  Block& implBlock = implOp.getBody().front();

  // create func.func private @eq(%self: !T, %other: !T) -> i1
  auto fnTy = builder.getFunctionType({receiverTy,receiverTy}, {builder.getI1Type()});
  auto eqFn = OpBuilder::atBlockBegin(&implBlock)
    .create<func::FuncOp>(loc, "eq", fnTy);
  eqFn.setPrivate();

  // build function body
  Block* entry = eqFn.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value self = entry->getArgument(0);
  Value other = entry->getArgument(1);

  // %res = tuple.cmp eq, %self, %other : !T
  auto cmpOp = builder.create<CmpOp>(
    loc,
    builder.getI1Type(),
    CmpPredicate::eq,
    self,
    other
  );

  // return %res : i1
  builder.create<func::ReturnOp>(
    loc,
    cmpOp.getResult()
  );

  return implOp;
}


struct CmpOpPartialEqLowering : OpRewritePattern<CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op,
                                PatternRewriter& rewriter) const override {
    // only lower predicates that use PartialEq
    if (op.getTraitName() != "PartialEq") {
      return rewriter.notifyMatchFailure(op, "trait name must be PartialEq");
    }

    // only lower concrete tuple types
    TupleType tupleTy = dyn_cast<TupleType>(op.getLhs().getType());
    if (!tupleTy || trait::containsSymbolicType(tupleTy)) {
      return rewriter.notifyMatchFailure(op, "tuple type must be concrete");
    }

    MLIRContext* ctx = op.getContext();
    auto loc = op.getLoc();

    // Before looping over tuple elements, generate the accumulator value
    // init = true for eq, false for ne
    auto predicate = op.getPredicate();
    Value acc = (predicate == CmpPredicate::eq)
      ? rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true))
      : rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));

    // if the tuple is empty, return early without needing to look up the trait
    if (tupleTy.getTypes().empty()) {
      rewriter.replaceOp(op, acc);
      return success();
    }

    // ensure that the trait.impl @PartialEq for `!tuple.any` exists
    //
    // XXX TODO it's unclear whether it's good to automatically generate
    //          this impl. It allows us to use tuple.cmp without an
    //          existing trait.impl @PartialEq for !tuple.any<[@PartialEq]>
    if (!getOrCreatePartialEqImpl(rewriter, op.getTrait())) {
      return rewriter.notifyMatchFailure(op, "couldn't get or create @PartialEq impl");
    }

    StringRef methodName = op.getMethodName();

    // build the method's function type
    // it's always the same, for any predicate:
    //
    // (!trait.self, !trait.self) -> i1
    Type selfTy = trait::SelfType::get(ctx);
    Type i1Ty = rewriter.getI1Type();
    FunctionType methodFunctionTy = FunctionType::get(ctx, {selfTy,selfTy}, {i1Ty});

    // loop over tuple elements
    for (auto [i, elemTy] : llvm::enumerate(tupleTy.getTypes())) {
      // extract element i from lhs & rhs
      Value lhs = rewriter.create<GetOp>(
        loc, elemTy, op.getLhs(), rewriter.getIndexAttr(i));
      Value rhs = rewriter.create<GetOp>(
        loc, elemTy, op.getRhs(), rewriter.getIndexAttr(i));

      // call @TraitName::@methodName<lhs_ty>(lhs, rhs)
      TypeAttr receiverTyAttr = TypeAttr::get(lhs.getType());
      
      auto call = rewriter.create<trait::MethodCallOp>(
        loc,
        /*results=*/TypeRange{i1Ty},
        "PartialEq",
        methodName,
        TypeAttr::get(methodFunctionTy),
        receiverTyAttr,
        /*operands=*/ValueRange{lhs,rhs}
      );
      Value cmp_i = call.getResult(0);

      if (predicate == CmpPredicate::eq) {
        // when predicate is eq, AND the comparison with the accumulator
        acc = rewriter.create<arith::AndIOp>(loc, acc, cmp_i);
      } else {
        // when predicate is ne, OR the comparison with the accumulator
        acc = rewriter.create<arith::OrIOp>(loc, acc, cmp_i);
      }
    }

    rewriter.replaceOp(op, acc);
    return success();
  }
};

void populateTupleToTraitConversionPatterns(RewritePatternSet& patterns) {
  patterns.add<CmpOpPartialEqLowering>(patterns.getContext());
}

} // end mlir::tuple
