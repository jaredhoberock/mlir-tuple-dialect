// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <memory>
#include <mlir/Pass/Pass.h>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace tuple {

// Registers the tuple-owned type conversions on an LLVM type converter: a
// tuple maps to the LLVM literal struct it lowers to, and a type that merely
// contains a tuple is rebuilt around the converted tuple. This is the one
// definition of the tuple->struct mapping; the data-layout model reuses it to
// recover the struct a tuple lowers to, so a tuple's layout and its lowering
// agree by construction.
void populateTupleToLLVMTypeConversions(LLVMTypeConverter& typeConverter);

void populateTupleToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                           RewritePatternSet& patterns);

struct ConvertTupleToLLVMPass
    : PassWrapper<ConvertTupleToLLVMPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTupleToLLVMPass);

  StringRef getArgument() const final { return "convert-tuple-to-llvm"; }
  StringRef getDescription() const final {
    return "Convert tuple dialect ops and tuple types to LLVM.";
  }

  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;
};

std::unique_ptr<Pass> createConvertTupleToLLVMPass();
}
}
