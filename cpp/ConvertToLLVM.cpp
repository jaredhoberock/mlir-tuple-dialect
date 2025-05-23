#include "ConvertToLLVM.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tuple {

static std::optional<int> getHomogeneousIntegerWidthOfPossibleIntTuple(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    return intTy.getWidth();
  }

  if (auto tupleTy = dyn_cast<TupleType>(ty)) {
    std::optional<int> width;

    if (tupleTy.getTypes().empty()) {
      // the tuple is vacuously homogeneous
      return 0;
    }

    for (Type elemTy : tupleTy.getTypes()) {
      auto elemWidth = getHomogeneousIntegerWidthOfPossibleIntTuple(elemTy);

      if (!elemWidth) {
        return std::nullopt;
      }

      if (*elemWidth == 0) {
        // vacuous cases don't participate in
        // determining the homogeneous integer width
        continue;
      }

      if (!width) {
        width = elemWidth;
      } else if (*width != *elemWidth) {
        return std::nullopt;
      }
    }

    // if all elements were empty, that's fine; return 0
    return width.value_or(0);
  }

  // not an integer or tuple
  return std::nullopt;
}

// we'll consider empty tuples to be int tuples
static bool isIntegerOrHomogeneousIntegerTuple(Type ty) {
  return getHomogeneousIntegerWidthOfPossibleIntTuple(ty).has_value();
}

static bool isHomogeneousIntegerTuple(TupleType tupleTy) {
  return isIntegerOrHomogeneousIntegerTuple(tupleTy);
}

// recursively count the number of integer leaves in an int tuple type
static int64_t countIntegerLeaves(Type ty) {
  if (ty.isInteger())
    return 1;

  if (auto tupleTy = dyn_cast<TupleType>(ty)) {
    int64_t total = 0;
    for (Type elementTy : tupleTy.getTypes()) {
      total += countIntegerLeaves(elementTy);
    }
    return total;
  }

  llvm_unreachable("countIntegerLeaves: unsupported type in int tuple");
}

struct IntTupleConstantOpLowering : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto tupleTy = cast<TupleType>(op.getResult().getType());
    std::optional<int> maybeIntWidth =
      getHomogeneousIntegerWidthOfPossibleIntTuple(tupleTy);

    // only lower homogeneous int tuples
    if (!maybeIntWidth) {
      return failure();
    }

    int intWidth = *maybeIntWidth;
    Location loc = op.getLoc();

    // count the number of integer leaves
    int64_t N = countIntegerLeaves(tupleTy);

    // XXX is it good to handle tuple<> in this matchAndRewrite?
    // special case: tuple<> lowers to i1
    if (N == 0) {
      Type i1Ty = rewriter.getIntegerType(1);
      Value zero = rewriter.create<arith::ConstantOp>(
        loc,
        i1Ty,
        rewriter.getIntegerAttr(i1Ty, 0)
      );
      rewriter.replaceOp(op, zero);
      return success();
    }

    // create result vector type
    auto intTy = rewriter.getIntegerType(intWidth);
    auto resultVectorTy = cast<VectorType>(getTypeConverter()->convertType(tupleTy));

    // begin from a single zero value across the entire vector
    auto zero = rewriter.create<arith::ConstantOp>(
      loc,
      intTy,
      rewriter.getIntegerAttr(intTy, 0)
    );
    Value result = rewriter.create<vector::SplatOp>(
      loc,
      resultVectorTy,
      zero
    );

    // insert operands into the result vector
    int64_t offset = 0;
    for (Value operand : adaptor.getOperands()) {
      // check the type of the lowered operand
      Type operandTy = operand.getType();

      if (!operandTy.isInteger(intWidth) && !operandTy.isInteger(1) &&
          !(isa<VectorType>(operandTy) && cast<VectorType>(operandTy).getElementType().isInteger(intWidth))) {
        return rewriter.notifyMatchFailure(op, "operand must be i1, iW, or vector<NxiW>");
      }

      // skip i1 (empty tuple) operands, they contribute no elements
      if (operandTy.isInteger(1)) {
        continue;
      }

      // if operand is a vector, use vector.insert_strided_slice
      if (auto vecTy = dyn_cast<VectorType>(operandTy)) {
        int64_t numElements = vecTy.getNumElements();
        SmallVector<int64_t> offsets = {offset};
        SmallVector<int64_t> strides = {1};

        result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, 
          operand, 
          result, 
          offsets, 
          strides
        );

        offset += numElements;
      } else {
        // For a scalar integer operand, use vector.insert
        result = rewriter.create<vector::InsertOp>(loc, operand, result, offset);
        offset += 1;
      }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GenericGetOpLowering : OpConversionPattern<GetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tupleTy = op.getTupleType();

    // only lower tuples which aren't homogeneous int tuples
    if (isHomogeneousIntegerTuple(tupleTy)) {
      return failure();
    }

    // XXX TODO we would lower this to some sort of extract-from-struct operation

    llvm_unreachable("ConstantOpLowering::matchAndRewrite: unimplemented");
    return failure();
  }
};

struct GenericConstantOpLowering : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type tupleTy = op.getResult().getType();

    // only lower tuples which aren't homogenous int tuples
    if (isIntegerOrHomogeneousIntegerTuple(tupleTy)) {
      return failure();
    }

    // XXX TODO we would lower this to an !llvm.struct

    llvm_unreachable("ConstantOpLowering::matchAndRewrite: unimplemented");
    return failure();
  }
};

struct IntTupleGetOpLowering : OpConversionPattern<GetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TupleType tupleTy = op.getTupleType();

    // only lower tuples which are homogeneous integer tuples
    if (!isHomogeneousIntegerTuple(tupleTy)) {
      return rewriter.notifyMatchFailure(op, "unsupported tuple element types");
    }

    Location loc = op.getLoc();
    int64_t index = op.getIndex().getSExtValue();
    Type elementTy = tupleTy.getType(index);
    Type loweredElementTy = getTypeConverter()->convertType(elementTy);

    // special case: if we're getting an empty tuple, materialize a constant zero i1
    if (loweredElementTy.isInteger(1)) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op,
        loweredElementTy,
        rewriter.getIntegerAttr(loweredElementTy, 0)
      );
      return success();
    }

    // compute flat offset to the element within the lowered vector
    int64_t offset = 0;
    for (int i = 0; i < index; ++i) {
      offset += countIntegerLeaves(elementTy);
    }

    Value loweredTupleVal = adaptor.getTuple();

    if (loweredElementTy.isInteger()) {
      // extract a single integer element from the vector
      rewriter.replaceOpWithNewOp<vector::ExtractOp>(
        op,
        loweredTupleVal,
        offset
      );
      return success();
    }

    if (auto vecTy = dyn_cast<VectorType>(loweredElementTy)) {
      // extract a contiguous slice from the vector
      SmallVector<int64_t> offsets = {offset};
      SmallVector<int64_t> sizes = {vecTy.getNumElements()};
      SmallVector<int64_t> strides = {1};

      rewriter.replaceOpWithNewOp<vector::ExtractStridedSliceOp>(
        op,
        loweredTupleVal,
        offsets,
        sizes,
        strides
      );
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported lowered result type");
  }
};

void populateTupleToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add a type conversion for TupleTypes that are homogeneous integer tuples
  typeConverter.addConversion([&](TupleType tupleTy) -> std::optional<Type> {
    // convert only TupleTypes that are homogeneous integer tuples
    if (!isHomogeneousIntegerTuple(tupleTy)) {
      // let other converters try
      return std::nullopt;
    }

    auto ctx = tupleTy.getContext();
    int64_t n = countIntegerLeaves(tupleTy);

    // 0-element vectors are illegal, so lower empty tuples to i1
    if (n == 0) return IntegerType::get(ctx, 1);

    // otherwise, use vector<nxiW>
    int intWidth = *getHomogeneousIntegerWidthOfPossibleIntTuple(tupleTy);
    return VectorType::get({n}, IntegerType::get(ctx, intWidth));
  });

  // XXX TODO add a type conversion for generic TupleType to !llvm.struct

  // lower ops
  patterns.add<
    IntTupleConstantOpLowering,
    GenericConstantOpLowering,
    IntTupleGetOpLowering,
    GenericGetOpLowering
  >(typeConverter, patterns.getContext());

  vector::populateVectorInsertExtractStridedSliceTransforms(patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
}

} // end mlir::tuple
