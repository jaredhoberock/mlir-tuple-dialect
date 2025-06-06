#include "ConvertToLLVM.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tuple {

struct AppendOpLowering : OpConversionPattern<AppendOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AppendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputTupleTy = dyn_cast_or_null<TupleType>(op.getTuple().getType());
    if (!inputTupleTy)
      return rewriter.notifyMatchFailure(op, "unsupported input tuple type");

    auto resultTupleTy = dyn_cast_or_null<TupleType>(op.getResult().getType());
    if (!resultTupleTy)
      return rewriter.notifyMatchFailure(op, "unsupported result tuple type");

    auto loc = op.getLoc();

    // get the lowered result struct type
    auto resultStructTy = cast<LLVM::LLVMStructType>(getTypeConverter()->convertType(resultTupleTy));

    // start with an undefined struct
    Value result = rewriter.create<LLVM::UndefOp>(loc, resultStructTy);

    // extract each field from the input tuple and insert into result
    Value inputStruct = adaptor.getTuple();
    for (auto idx : llvm::seq<unsigned>(0, inputTupleTy.size())) {
      Value field = rewriter.create<LLVM::ExtractValueOp>(
        loc, inputStruct, idx);

      result = rewriter.create<LLVM::InsertValueOp>(
        loc, result, field, idx);
    }

    // insert the appended element as the last field
    result = rewriter.create<LLVM::InsertValueOp>(
      loc, result, adaptor.getElement(), inputTupleTy.size());

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConstantOpLowering : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tupleTy = cast<TupleType>(op.getResult().getType());
    Location loc = op.getLoc();

    // handle empty tuples
    if (tupleTy.getTypes().empty()) {
      auto emptyStructTy = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        {}
      );

      Value emptyStruct = rewriter.create<LLVM::UndefOp>(loc, emptyStructTy);
      rewriter.replaceOp(op, emptyStruct);
      return success();
    }

    // get the lowered struct type
    auto structTy = cast<LLVM::LLVMStructType>(getTypeConverter()->convertType(tupleTy));

    // start with an undefined struct
    Value result = rewriter.create<LLVM::UndefOp>(loc, structTy);

    // insert each operand into the struct
    for (auto [idx, operand] : llvm::enumerate(adaptor.getOperands())) {
      result = rewriter.create<LLVM::InsertValueOp>(
        loc,
        result,
        operand,
        idx
      );
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GetOpLowering : OpConversionPattern<GetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t index = op.getIndex().getSExtValue();

    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
      op,
      adaptor.getTuple(),
      index
    );

    return success();
  }
};

void populateTupleToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add a type conversion all TupleTypes to !llvm.struct
  typeConverter.addConversion([&](TupleType tupleTy) -> std::optional<Type> {
    SmallVector<Type> elementTypes;

    // convert each element type
    for (Type elemTy : tupleTy.getTypes()) {
      Type convertedElemTy = typeConverter.convertType(elemTy);
      if (!convertedElemTy) {
        return std::nullopt;
      }
      elementTypes.push_back(convertedElemTy);
    }

    // create LLVM struct
    return LLVM::LLVMStructType::getLiteral(tupleTy.getContext(), elementTypes);
  });

  // add LLVM-specific lowering patterns
  patterns.add<
    AppendOpLowering,
    ConstantOpLowering,
    GetOpLowering
  >(typeConverter, patterns.getContext());
}

} // end mlir::tuple
