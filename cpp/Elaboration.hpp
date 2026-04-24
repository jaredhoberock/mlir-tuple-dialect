// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

class RewritePatternSet;

namespace tuple {

/// Elaborates higher-level tuple operations into the `tuple.make` /
/// `tuple.get` core. Operates entirely within the tuple dialect on
/// monomorphic input; does not introduce or reason about trait IR.
void populateTupleElaborationPatterns(RewritePatternSet& patterns);

struct TupleElaboratePass : PassWrapper<TupleElaboratePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TupleElaboratePass);

  StringRef getArgument() const final { return "tuple-elaborate"; }
  StringRef getDescription() const final {
    return "Elaborate higher-level tuple ops into the tuple.make/tuple.get core.";
  }

  void runOnOperation() override;
};

} // end tuple
} // end mlir
