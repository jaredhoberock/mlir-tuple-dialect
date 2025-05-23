#include "ConvertToTrait.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace mlir::tuple {

static func::FuncOp buildImplMethod(
    OpBuilder& builder,
    Location loc,
    Type receiverTy,
    StringRef methodName,
    CmpPredicate predicate) {
  PatternRewriter::InsertionGuard guard(builder);

  // create this op:
  //
  // func.func private @methodName(%self: !T, %other: %!T) -> i1 {
  //   %res = tuple.cmp predicate, %self, %other : !T
  //   return %res : i1
  // }

  // create func.func private @methodName(%self: !T, %other: !T) -> i1
  auto fnTy = builder.getFunctionType({receiverTy,receiverTy}, {builder.getI1Type()});
  auto funcOp = builder.create<func::FuncOp>(loc, methodName, fnTy);
  funcOp.setPrivate();

  // build function body
  Block* entry = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value self = entry->getArgument(0);
  Value other = entry->getArgument(1);

  // %res = tuple.cmp predicate, %self, %other : !T
  auto cmpOp = builder.create<CmpOp>(
    loc,
    builder.getI1Type(),
    predicate,
    self,
    other
  );

  // return %res : i1
  builder.create<func::ReturnOp>(
    loc,
    cmpOp.getResult()
  );

  return funcOp;
}

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

  // build @eq
  builder.setInsertionPointToStart(&implBlock);
  buildImplMethod(builder, loc, receiverTy, "eq", CmpPredicate::eq);

  return implOp;
}

static trait::ImplOp getOrCreatePartialOrdImpl(OpBuilder& builder, trait::TraitOp partialOrdOp) {
  FlatSymbolRefAttr partialOrdRefAttr =
    FlatSymbolRefAttr::get(builder.getContext(), "PartialOrd");

  // construct the receiver type
  // !T = !tuple.any<[@PartialOrd]>
  Type receiverTy =
    AnyTupleType::get(builder.getContext(), {partialOrdRefAttr});

  // check if the impl already exists for this receiver
  if (auto existing = partialOrdOp.getImpl(receiverTy)) return existing;

  // the impl doesn't exist yet; create it
  PatternRewriter::InsertionGuard guard(builder);

  MLIRContext* ctx = builder.getContext();
  Location loc = partialOrdOp.getLoc();

  // create trait.impl @PartialOrd for !T { ... }
  builder.setInsertionPointAfter(partialOrdOp);
  auto implOp = builder.create<trait::ImplOp>(
    loc,
    partialOrdRefAttr,
    TypeAttr::get(receiverTy)
  );

  // get the pre-created entry block
  Block& implBlock = implOp.getBody().front();

  // build the methods the trait says we need
  builder.setInsertionPointToStart(&implBlock);
  buildImplMethod(builder, loc, receiverTy, "lt", CmpPredicate::lt);
  buildImplMethod(builder, loc, receiverTy, "le", CmpPredicate::le);
  buildImplMethod(builder, loc, receiverTy, "gt", CmpPredicate::gt);
  buildImplMethod(builder, loc, receiverTy, "ge", CmpPredicate::ge);

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
      return rewriter.notifyMatchFailure(op, "couldn't get or create PartialEq impl");
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
        i1Ty,
        "PartialEq",
        methodName,
        TypeAttr::get(methodFunctionTy),
        receiverTyAttr,
        ValueRange{lhs,rhs}
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


// Pattern for strict comparisons (lt, gt)
struct CmpOpPartialOrdStrictLowering : OpRewritePattern<CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op, PatternRewriter &rewriter) const override {
    if (op.getTraitName() != "PartialOrd")
      return rewriter.notifyMatchFailure(op, "trait name must be PartialOrd");
    
    StringRef methodName = op.getMethodName();
    if (methodName != "lt" && methodName != "gt")
      return rewriter.notifyMatchFailure(op, "only handles lt/gt");

    auto tupleTy = dyn_cast<TupleType>(op.getLhs().getType());
    if (!tupleTy || trait::containsSymbolicType(tupleTy))
      return rewriter.notifyMatchFailure(op, "tuple must be concrete");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Empty tuples are always false for strict comparisons
    if (tupleTy.getTypes().empty()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, rewriter.getBoolAttr(false));
      return success();
    }

    if (!getOrCreatePartialOrdImpl(rewriter, op.getTrait())) {
      return rewriter.notifyMatchFailure(op, "could not create PartialOrd impl");
    }

    Type selfTy = trait::SelfType::get(ctx);
    Type i1Ty = rewriter.getI1Type();
    FunctionType methodFunctionTy = FunctionType::get(ctx, {selfTy,selfTy}, {i1Ty});

    Value result = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value allPrevEqual = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));

    for (auto [i, elemTy] : llvm::enumerate(tupleTy.getTypes())) {
      auto idx = rewriter.getIndexAttr(i);
      Value lhs_i = rewriter.create<GetOp>(loc, elemTy, op.getLhs(), idx);
      Value rhs_i = rewriter.create<GetOp>(loc, elemTy, op.getRhs(), idx);
      TypeAttr receiverTy = TypeAttr::get(elemTy);

      // Call the method
      Value cmp_i = rewriter.create<trait::MethodCallOp>(
        loc, i1Ty, "PartialOrd", methodName,
        TypeAttr::get(methodFunctionTy), receiverTy,
        ValueRange{lhs_i, rhs_i}
      ).getResult(0);

      // Contribute if all previous were equal and this is true
      Value contribution = rewriter.create<arith::AndIOp>(loc, allPrevEqual, cmp_i);
      result = rewriter.create<arith::OrIOp>(loc, result, contribution);

      // Check equality for next iteration
      Value cmp_rev = rewriter.create<trait::MethodCallOp>(
        loc, i1Ty, "PartialOrd", methodName,
        TypeAttr::get(methodFunctionTy), receiverTy,
        ValueRange{rhs_i, lhs_i}
      ).getResult(0);

      // Equal if neither direction is true
      Value either = rewriter.create<arith::OrIOp>(loc, cmp_i, cmp_rev);
      Value constTrue = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      Value eq_i = rewriter.create<arith::XOrIOp>(loc, either, constTrue);
      
      allPrevEqual = rewriter.create<arith::AndIOp>(loc, allPrevEqual, eq_i);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

// Pattern for non-strict comparisons (le, ge)
struct CmpOpPartialOrdNonStrictLowering : OpRewritePattern<CmpOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpOp op, PatternRewriter &rewriter) const override {
    if (op.getTraitName() != "PartialOrd")
      return rewriter.notifyMatchFailure(op, "trait name must be PartialOrd");
    
    StringRef methodName = op.getMethodName();
    if (methodName != "le" && methodName != "ge")
      return rewriter.notifyMatchFailure(op, "only handles le/ge");

    auto tupleTy = dyn_cast<TupleType>(op.getLhs().getType());
    if (!tupleTy || trait::containsSymbolicType(tupleTy))
      return rewriter.notifyMatchFailure(op, "tuple must be concrete");

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Empty tuples are always true for non-strict comparisons
    if (tupleTy.getTypes().empty()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, rewriter.getBoolAttr(true));
      return success();
    }

    if (!getOrCreatePartialOrdImpl(rewriter, op.getTrait())) {
      return rewriter.notifyMatchFailure(op, "could not create PartialOrd impl");
    }

    Type selfTy = trait::SelfType::get(ctx);
    Type i1Ty = rewriter.getI1Type();
    FunctionType methodFunctionTy = FunctionType::get(ctx, {selfTy,selfTy}, {i1Ty});

    Value result = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value allPrevEqual = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));

    for (auto [i, elemTy] : llvm::enumerate(tupleTy.getTypes())) {
      auto idx = rewriter.getIndexAttr(i);
      Value lhs_i = rewriter.create<GetOp>(loc, elemTy, op.getLhs(), idx);
      Value rhs_i = rewriter.create<GetOp>(loc, elemTy, op.getRhs(), idx);
      TypeAttr receiverTy = TypeAttr::get(elemTy);

      // Call the method
      Value cmp_i = rewriter.create<trait::MethodCallOp>(
        loc, i1Ty, "PartialOrd", methodName,
        TypeAttr::get(methodFunctionTy), receiverTy,
        ValueRange{lhs_i, rhs_i}
      ).getResult(0);

      // Check equality
      Value cmp_rev = rewriter.create<trait::MethodCallOp>(
        loc, i1Ty, "PartialOrd", methodName,
        TypeAttr::get(methodFunctionTy), receiverTy,
        ValueRange{rhs_i, lhs_i}
      ).getResult(0);
      
      Value eq_i = rewriter.create<arith::AndIOp>(loc, cmp_i, cmp_rev);

      // Only contribute if strictly less/greater (not equal)
      Value constTrue = rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      Value not_eq_ = rewriter.create<arith::XOrIOp>(loc, eq_i, constTrue);
      Value strict_cmp = rewriter.create<arith::AndIOp>(loc, cmp_i, not_eq_);
      
      Value contribution = rewriter.create<arith::AndIOp>(loc, allPrevEqual, strict_cmp);
      result = rewriter.create<arith::OrIOp>(loc, result, contribution);
      
      allPrevEqual = rewriter.create<arith::AndIOp>(loc, allPrevEqual, eq_i);
    }

    // For le/ge, also true if all elements are equal
    result = rewriter.create<arith::OrIOp>(loc, result, allPrevEqual);

    rewriter.replaceOp(op, result);
    return success();
  }
};

void populateTupleToTraitConversionPatterns(RewritePatternSet& patterns) {
  patterns.add<
    CmpOpPartialEqLowering,
    CmpOpPartialOrdNonStrictLowering,
    CmpOpPartialOrdStrictLowering
  >(patterns.getContext());
}

} // end mlir::tuple
