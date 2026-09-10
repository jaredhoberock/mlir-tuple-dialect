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

struct TupleInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // Every tuple op may be inlined anywhere: the dialect's ops carry no state
  // that forbids cloning, and the compiler's inliner clones a callee only when
  // every op in it belongs to a dialect declaring inlining legal, so a host
  // function reading a tuple is inlinable only once every tuple op is.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  // A region may be inlined into a tuple region.
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
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
    TupleInlinerInterface
  >();
}

void TupleDialect::getCanonicalizationPatterns(RewritePatternSet& patterns) const {
  populateTupleCanonicalizationPatterns(patterns);
}

/// Answers a decoded tuple value with a `tuple.constant`: the value the bind
/// hands here is the tuple's nested attribute, an array attribute with one entry
/// per element. Any other attribute or a non-tuple type is not this dialect's to
/// materialize.
Operation *TupleDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  auto tupleTy = dyn_cast<TupleType>(type);
  auto array = dyn_cast<ArrayAttr>(value);
  if (!tupleTy || !array)
    return nullptr;
  return ConstantOp::create(builder, loc, tupleTy, array);
}

}
