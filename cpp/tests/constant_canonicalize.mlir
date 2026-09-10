// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(tuple-canonicalize)" %s | FileCheck %s

// tuple.make of all-constant operands folds to a tuple.constant gathering their
// attributes; a nested constant tuple contributes its array attribute.
// CHECK-LABEL: func.func @make_of_constants
// CHECK-NOT: tuple.make
// CHECK: tuple.constant([3, 4]) : tuple<i64, i64>
func.func @make_of_constants() -> tuple<i64, i64> {
  %a = arith.constant 3 : i64
  %b = arith.constant 4 : i64
  %t = tuple.make(%a, %b : i64, i64) : tuple<i64, i64>
  return %t : tuple<i64, i64>
}

// CHECK-LABEL: func.func @make_of_constant_tuple
// CHECK-NOT: tuple.make
// CHECK: tuple.constant([5, [6, 7]]) : tuple<i64, tuple<i64, i64>>
func.func @make_of_constant_tuple() -> tuple<i64, tuple<i64, i64>> {
  %a = arith.constant 5 : i64
  %inner = tuple.constant([6 : i64, 7 : i64]) : tuple<i64, i64>
  %t = tuple.make(%a, %inner : i64, tuple<i64, i64>) : tuple<i64, tuple<i64, i64>>
  return %t : tuple<i64, tuple<i64, i64>>
}

// tuple.get of a tuple.constant folds to a constant of the selected element.
// CHECK-LABEL: func.func @get_of_constant_scalar
// CHECK-NOT: tuple.get
// CHECK: %[[C:.*]] = arith.constant 9 : i32
// CHECK: return %[[C]] : i32
func.func @get_of_constant_scalar() -> i32 {
  %c = tuple.constant([7 : i32, 9 : i32]) : tuple<i32, i32>
  %e = tuple.get %c, 1 : tuple<i32, i32> -> i32
  return %e : i32
}

// The selected element of a nested constant is itself a tuple.constant.
// CHECK-LABEL: func.func @get_of_constant_tuple
// CHECK-NOT: tuple.get
// CHECK: tuple.constant([6, 7]) : tuple<i64, i64>
func.func @get_of_constant_tuple() -> tuple<i64, i64> {
  %c = tuple.constant([5 : i64, [6 : i64, 7 : i64]]) : tuple<i64, tuple<i64, i64>>
  %e = tuple.get %c, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
  return %e : tuple<i64, i64>
}
