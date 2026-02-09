// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Canonicalization.hpp"
#include "ConvertToLLVM.hpp"
#include "ImplGenerators.hpp"
#include "Monomorphization.hpp"
#include "Tuple.hpp"
#include "TupleOps.hpp"
#include <llvm/ADT/STLExtras.h>
#include <iostream>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Transforms/InliningUtils.h>
#include <Trait.hpp>

#include "Tuple.cpp.inc"

namespace mlir::tuple {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populateTupleToLLVMConversionPatterns(typeConverter, patterns);
  }
};

struct MonomorphizationInterface : public trait::MonomorphizationInterface {
  using trait::MonomorphizationInterface::MonomorphizationInterface;

  void populateConvertToTraitPatterns(RewritePatternSet& patterns) const override final {
    tuple::populateConvertTupleToTraitPatterns(patterns);
  }

  void populateInstantiateMonomorphsPatterns(RewritePatternSet& patterns) const override final {
    tuple::populateInstantiateMonomorphsPatterns(patterns);
  }

  void populateEraseClaimsPatterns(TypeConverter &typeConverter,
                                   RewritePatternSet &patterns) const override final {
    tuple::populateEraseClaimsPatterns(typeConverter, patterns);
  }
};

struct GenerateImplsInterface : public trait::GenerateImplsInterface {
  using trait::GenerateImplsInterface::GenerateImplsInterface;

  void populateImplGenerators(trait::ImplGeneratorSet& generators) const override final {
    tuple::populateImplGenerators(generators);
  }
};

struct TupleInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  static bool isAlwaysLegalToInline(Operation *op) {
    // for now, assume tuple.append, tuple.get, & tuple.make are always safe to inline
    return isa<AppendOp>(op) || isa<GetOp>(op) || isa<MakeOp>(op);
  }

  bool isLegalToInline(Operation* op, Region*, bool, IRMapping&) const {
    return isAlwaysLegalToInline(op);
  }
};

void TupleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TupleOps.cpp.inc"
  >();

  registerTypes();

  addInterfaces<
    ConvertToLLVMInterface,
    GenerateImplsInterface,
    MonomorphizationInterface,
    TupleInlinerInterface
  >();
}

void TupleDialect::getCanonicalizationPatterns(RewritePatternSet& patterns) const {
  populateTupleCanonicalizationPatterns(patterns);
}

}
