// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Elaboration.hpp"
#include "Canonicalization.hpp"
#include "ConvertToLLVM.hpp"
#include "Tuple.hpp"
#include <mlir/Tools/Plugins/DialectPlugin.h>

static void registerPlugin(mlir::DialectRegistry* registry) {
  registry->insert<mlir::tuple::TupleDialect>();
  mlir::PassRegistration<mlir::tuple::TupleCanonicalizePass>();
  mlir::PassRegistration<mlir::tuple::ConvertTupleToLLVMPass>();
  mlir::PassRegistration<mlir::tuple::TupleElaboratePass>();
}

extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "TupleDialectPlugin",
    "v0.1",
    registerPlugin
  };
}
