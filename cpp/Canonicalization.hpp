// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

namespace tuple {

void populateTupleCanonicalizationPatterns(RewritePatternSet& patterns);

struct TupleCanonicalizePass
    : PassWrapper<TupleCanonicalizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TupleCanonicalizePass);

  StringRef getArgument() const final { return "tuple-canonicalize"; }
  StringRef getDescription() const final {
    return "Apply focused canonicalization patterns for tuple dialect ops.";
  }

  void runOnOperation() override;
};

} // end tuple
} // end mlir
