// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// The unit type (the empty tuple, an empty aggregate) carries no data. The exact
// model is the zero-field LLVM struct, but the NVPTX backend rejects an empty
// aggregate as a kernel parameter, and a unit reaches a device kernel as its
// captured-argument type whenever a launch captures nothing. So the unit lowers
// to a single byte, and that byte is a defined zero rather than undef: a
// specialized unit then writes a deterministic value and its specialization key
// is stable. A unit make and a unit constant both produce that defined zero.

// -----
// CHECK-LABEL: llvm.func @unit_make
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.mlir.constant(0 : i8) : i8
// CHECK: llvm.return
func.func @unit_make() -> tuple<> {
  %c = tuple.make : tuple<>
  return %c : tuple<>
}

// -----
// CHECK-LABEL: llvm.func @unit_constant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.mlir.constant(0 : i8) : i8
// CHECK: llvm.return
func.func @unit_constant() -> tuple<> {
  %c = tuple.constant([]) : tuple<>
  return %c : tuple<>
}
