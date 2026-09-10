// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// A tuple.constant whose element type is a non-tuple aggregate carrying an array
// leaf has no scalar the struct builder can place: the lowering fails the match
// rather than aborting, so the op survives the partial conversion.
// CHECK-LABEL: llvm.func @t
// CHECK: tuple.constant
func.func @t() -> tuple<!llvm.struct<(i64)>> {
  %c = tuple.constant([[1 : i64]]) : tuple<!llvm.struct<(i64)>>
  return %c : tuple<!llvm.struct<(i64)>>
}
