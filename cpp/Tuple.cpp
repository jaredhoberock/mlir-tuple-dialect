// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Canonicalization.hpp"
#include "ConvertToLLVM.hpp"
#include "ImplGenerators.hpp"
#include "Monomorphization.hpp"
#include "PermissiveInlinerInterface.hpp"
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
#include <ConstantValue.hpp>
#include <Trait.hpp>

#include <Tuple.cpp.inc>

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

  void populateErasePolymorphsPatterns(TypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      AttrTypeReplacer &) const override final {
    tuple::populateErasePolymorphsPatterns(typeConverter, patterns);
  }
};

struct GenerateImplsInterface : public trait::GenerateImplsInterface {
  using trait::GenerateImplsInterface::GenerateImplsInterface;

  void populateImplGenerators(trait::ImplGeneratorSet& generators) const override final {
    tuple::populateImplGenerators(generators);
  }
};

void TupleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include <TupleOps.cpp.inc>
  >();

  registerTypes();
  registerTupleDataLayoutInterface(getContext());

  addInterfaces<
    ConvertToLLVMInterface,
    GenerateImplsInterface,
    MonomorphizationInterface,
    lowering::PermissiveInlinerInterface<TupleDialect>
  >();

}

void TupleDialect::getCanonicalizationPatterns(RewritePatternSet& patterns) const {
  populateTupleCanonicalizationPatterns(patterns);
}

/// Answers a decoded value with the constant op that stands for it: a tuple type
/// with a matching-arity array attribute is a `tuple.constant`; every other type
/// is a leaf this dialect owns no constant op for. A leaf hook: a tuple-owned type
/// that is not a `TupleType` answers null, as does each arity mismatch, leaving
/// the value unmaterialized.
Operation *TupleDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto tupleTy = dyn_cast<TupleType>(type)) {
    auto array = dyn_cast<ArrayAttr>(value);
    if (!array || array.size() != tupleTy.size())
      return nullptr;
    return ConstantOp::create(builder, loc, tupleTy, array);
  }
  return lowering::materializeConstantAsLeaf(this, builder, value, type, loc);
}

}
