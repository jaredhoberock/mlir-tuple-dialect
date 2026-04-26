// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "ConvertToLLVM.hpp"
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include <TraitTypes.hpp>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/Transforms/Patterns.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tuple {

/// %field = tuple.get %tuple[INDEX] : tuple<Ts...>
///   -> %field = llvm.extractvalue %tuple[INDEX] : !llvm.struct<...>
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

/// %tuple = tuple.make %a, %b, ... : tuple<A, B, ...>
///   -> %tuple = llvm.mlir.undef : !llvm.struct<(A', B', ...)>
///      %tuple0 = llvm.insertvalue %a, %tuple[0] : !llvm.struct<(A', B', ...)>
///      %tuple1 = llvm.insertvalue %b, %tuple0[1] : !llvm.struct<(A', B', ...)>
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

struct ConvertAnyOpWithTupleTypes : public ConversionPattern {
  ConvertAnyOpWithTupleTypes(const TypeConverter &tc, MLIRContext *ctx)
      : ConversionPattern(tc, Pattern::MatchAnyOpTypeTag(), /*benefit=*/1,
                          ctx) {}

  /// Rebuilds non-tuple ops whose boundary mentions TupleType, preserving
  /// attributes, properties, successors, and regions while converting types.
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getDialect() && isa<TupleDialect>(op->getDialect()))
      return failure();
    if (!trait::opMentionsType<TupleType>(op))
      return failure();
    if (getTypeConverter()->isLegal(op))
      return failure();

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes)))
      return rewriter.notifyMatchFailure(op, "could not convert result types");

    IRMapping mapper;
    for (auto [oldOperand, newOperand] : llvm::zip(op->getOperands(), operands))
      mapper.map(oldOperand, newOperand);

    Operation *newOp = Operation::create(
        op->getLoc(), op->getName(), newResultTypes, operands, op->getAttrs(),
        op->getPropertiesStorage(), op->getSuccessors(), op->getNumRegions());
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults()))
      mapper.map(oldResult, newResult);

    rewriter.insert(newOp);
    for (auto [oldRegion, newRegion] :
         llvm::zip(op->getRegions(), newOp->getRegions())) {
      oldRegion.cloneInto(&newRegion, mapper);
      if (failed(rewriter.convertRegionTypes(&newRegion, *getTypeConverter()))) {
        rewriter.eraseOp(newOp);
        return rewriter.notifyMatchFailure(op, "could not convert region types");
      }
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

/// Adds only tuple-owned type conversions. TupleType lowers to an LLVM literal
/// struct; other structural types are rebuilt only when they contain TupleType.
static void populateTupleToLLVMTypeConversions(LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](Type type) -> std::optional<Type> {
    if (isa<TupleType>(type))
      return std::nullopt;
    if (!trait::containsType<TupleType>(type))
      return std::nullopt;

    SmallVector<Attribute> subAttrs;
    SmallVector<Type> subTypes;
    type.walkImmediateSubElements([&](Attribute attr) {
      subAttrs.push_back(attr);
    }, [&](Type subType) {
      subTypes.push_back(subType);
    });

    bool changed = false;
    SmallVector<Type> newSubTypes;
    newSubTypes.reserve(subTypes.size());
    for (Type subType : subTypes) {
      Type converted = typeConverter.convertType(subType);
      if (!converted)
        return std::nullopt;
      changed |= converted != subType;
      newSubTypes.push_back(converted);
    }

    if (!changed)
      return type;
    return type.replaceImmediateSubElements(subAttrs, newSubTypes);
  });

  typeConverter.addConversion([&](TupleType tupleTy) -> std::optional<Type> {
    if (tupleTy.getTypes().empty())
      return IntegerType::get(tupleTy.getContext(), 8);

    SmallVector<Type> elementTypes;
    for (Type elemTy : tupleTy.getTypes()) {
      Type converted = typeConverter.convertType(elemTy);
      if (!converted)
        return std::nullopt;
      elementTypes.push_back(converted);
    }

    return LLVM::LLVMStructType::getLiteral(tupleTy.getContext(), elementTypes);
  });
}

/// Populates the conversion hook used by the tuple dialect's ConvertToLLVM
/// interface without making unrelated types legal.
void populateTupleToLLVMConversionPatterns(LLVMTypeConverter& typeConverter, RewritePatternSet& patterns) {
  populateTupleToLLVMTypeConversions(typeConverter);
  patterns.add<
    GetOpLowering,
    MakeOpLowering,
    ConvertAnyOpWithTupleTypes
  >(typeConverter, patterns.getContext());
}

void ConvertTupleToLLVMPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
}

void ConvertTupleToLLVMPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  LLVMTypeConverter typeConverter(ctx);

  // The standalone pass only owns tuple conversion, so foreign/unrelated types
  // remain legal here. The reusable hook must not install this fallback.
  typeConverter.addConversion([](Type type) { return type; });
  RewritePatternSet patterns(ctx);
  ConversionTarget target(*ctx);

  populateTupleToLLVMConversionPatterns(typeConverter, patterns);

  target.addIllegalDialect<TupleDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  auto opIsTupleLegal = [&](Operation *op) {
    if (!trait::opMentionsType<TupleType>(op))
      return true;

    if (auto funcOp = dyn_cast<FunctionOpInterface>(op))
      if (auto type = dyn_cast<FunctionType>(funcOp.getFunctionType()))
        return typeConverter.isSignatureLegal(type);

    return typeConverter.isLegal(op);
  };

  target.markUnknownOpDynamicallyLegal(opIsTupleLegal);

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
