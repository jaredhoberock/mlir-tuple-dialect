#include "Canonicalization.hpp"
#include "ConvertToLLVM.hpp"
#include "ConvertToTrait.hpp"
#include "Dialect.hpp"
#include "Ops.hpp"
#include <llvm/ADT/STLExtras.h>
#include <iostream>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir-trait-dialect/cpp/Dialect.hpp>

#include "Dialect.cpp.inc"

namespace mlir::tuple {

struct ConvertToLLVMInterface : public mlir::ConvertToLLVMPatternInterface {
  using mlir::ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void populateConvertToLLVMConversionPatterns(ConversionTarget& target,
                                               LLVMTypeConverter& typeConverter,
                                               RewritePatternSet& patterns) const override final {
    populateTupleToLLVMConversionPatterns(typeConverter, patterns);
  }
};

struct ConvertToTraitInterface : public mlir::trait::ConvertToTraitPatternInterface {
  using mlir::trait::ConvertToTraitPatternInterface::ConvertToTraitPatternInterface;

  void populateConvertToTraitConversionPatterns(RewritePatternSet& patterns) const override final {
    populateTupleToTraitConversionPatterns(patterns);
  }
};

void TupleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
  >();

  registerTypes();

  addInterfaces<
    ConvertToLLVMInterface,
    ConvertToTraitInterface
  >();
}

void TupleDialect::getCanonicalizationPatterns(RewritePatternSet& patterns) const {
  populateTupleCanonicalizationPatterns(patterns);
}

}
