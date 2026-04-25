// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "ConvertToLLVM.hpp"
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tuple {

struct GetOpLowering : OpConversionPattern<GetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t index = op.getIndex().getSExtValue();

    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
      op,
      adaptor.getTuple(),
      index
    );

    return success();
  }
};

struct MakeOpLowering : OpConversionPattern<MakeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MakeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tupleTy = cast<TupleType>(op.getResult().getType());
    Location loc = op.getLoc();

    // handle empty tuples
    if (tupleTy.getTypes().empty()) {
      Type convertedTy = getTypeConverter()->convertType(tupleTy);
      Value undefined = LLVM::UndefOp::create(rewriter, loc, convertedTy);
      rewriter.replaceOp(op, undefined);
      return success();
    }

    // get the lowered struct type
    auto structTy = cast<LLVM::LLVMStructType>(getTypeConverter()->convertType(tupleTy));

    // start with an undefined struct
    Value result = LLVM::UndefOp::create(rewriter, loc, structTy);

    // insert each operand into the struct
    for (auto [idx, operand] : llvm::enumerate(adaptor.getOperands())) {
      result = LLVM::InsertValueOp::create(rewriter, 
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

void populateTupleToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  // add a type conversion all TupleTypes to !llvm.struct
  typeConverter.addConversion([&](TupleType tupleTy) -> std::optional<Type> {
    // lower tuple<> directly to i8
    if (tupleTy.getTypes().empty()) {
      return IntegerType::get(tupleTy.getContext(), 8);
    }

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
    GetOpLowering,
    MakeOpLowering
  >(typeConverter, patterns.getContext());
}

void ConvertTupleToLLVMPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
}

void ConvertTupleToLLVMPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  LLVMTypeConverter typeConverter(ctx);
  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);

  populateTupleToLLVMConversionPatterns(typeConverter, patterns);

  target.addIllegalDialect<TupleDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return typeConverter.isLegal(op.getOperandTypes());
  });

  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                       patterns, target);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  if (failed(applyPartialConversion(getOperation(), target,
                                    std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> createConvertTupleToLLVMPass() {
  return std::make_unique<ConvertTupleToLLVMPass>();
}

} // end mlir::tuple
