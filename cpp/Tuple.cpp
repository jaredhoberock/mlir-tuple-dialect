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

  // A callee may be inlined into a tuple region only when it is a single block:
  // every tuple op's region is SizedRegion<1>, so splicing a multi-block callee
  // in would leave a region that fails to verify. (Upstream affine, whose
  // regions are likewise single-block, refuses on the same ground.)
  bool isLegalToInline(Region *, Region *src, bool, IRMapping &) const final {
    return src->hasOneBlock();
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

/// Answers a decoded value with the constant op that stands for it: a tuple type
/// with a matching-arity array attribute is a `tuple.constant`; a builtin scalar
/// is arith's constant; any other type's constant is its own dialect's. Each
/// mismatch answers null so the folder reports "no dialect materializes" rather
/// than minting an op that fails verification.
Operation *TupleDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto tupleTy = dyn_cast<TupleType>(type)) {
    auto array = dyn_cast<ArrayAttr>(value);
    if (!array || array.size() != tupleTy.size())
      return nullptr;
    return ConstantOp::create(builder, loc, tupleTy, array);
  }
  if (arith::ConstantOp c = arith::ConstantOp::materialize(builder, value, type, loc))
    return c.getOperation();
  return type.getDialect().materializeConstant(builder, value, type, loc);
}

}
