// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// A tuple.constant lowers to the LLVM struct value the make of those constants
// would lower to: an undefined struct with each leaf inserted.

// -----
// CHECK-LABEL: llvm.func @constant_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.mlir.constant(7 : i32)
// CHECK: llvm.insertvalue
// CHECK: llvm.mlir.constant(1.500000e+00 : f32)
// CHECK: llvm.insertvalue
// CHECK: llvm.return
func.func @constant_pair() -> tuple<i32, f32> {
  %c = tuple.constant([7 : i32, 1.5 : f32]) : tuple<i32, f32>
  return %c : tuple<i32, f32>
}

// -----
// A nested constant lowers to a nested struct value, no residual tuple ops.
// CHECK-LABEL: llvm.func @constant_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.mlir.undef : !llvm.struct<(i64, struct<(i32, f32)>)>
// CHECK: llvm.return
func.func @constant_nested() -> tuple<i64, tuple<i32, f32>> {
  %c = tuple.constant([42 : i64, [7 : i32, 1.5 : f32]]) : tuple<i64, tuple<i32, f32>>
  return %c : tuple<i64, tuple<i32, f32>>
}

// -----
// An index leaf lowers to the lowered integer the type converter chooses.
// CHECK-LABEL: llvm.func @constant_index
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.mlir.constant(5 : i64) : i64
// CHECK: llvm.return
func.func @constant_index() -> tuple<index> {
  %c = tuple.constant([5 : index]) : tuple<index>
  return %c : tuple<index>
}

// -----
// The empty tuple lowers to the defined zero byte its make lowers to.
// CHECK-LABEL: llvm.func @constant_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.mlir.constant(0 : i8) : i8
// CHECK: llvm.return
func.func @constant_empty() -> tuple<> {
  %c = tuple.constant([]) : tuple<>
  return %c : tuple<>
}
