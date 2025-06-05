#include "ConvertToLLVM.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tuple {

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
    ConstantOpLowering,
    GetOpLowering
  >(typeConverter, patterns.getContext());
}

} // end mlir::tuple
