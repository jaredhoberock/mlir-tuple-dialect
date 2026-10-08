// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// --tuple-elaborate rewrites only the higher-level tuple ops: IR that already
// contains only the tuple.make / tuple.get core stands, except that a tuple.get
// of a tuple.make folds to the operand it projects, the op's own fold, which
// every greedy rewrite applies. Covers several structural shapes so that any
// other rewrite on make/get would trip at least one case.

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(tuple-elaborate)" | FileCheck %s

// -----
// simple get-then-singleton-make

// CHECK-LABEL: func.func @noop_make_get
// CHECK: %[[G:.+]] = tuple.get %arg0, 0 : tuple<i32, i64> -> i32
// CHECK: %[[M:.+]] = tuple.make(%[[G]] : i32) : tuple<i32>
// CHECK: return %[[M]] : tuple<i32>
func.func @noop_make_get(%t: tuple<i32, i64>) -> tuple<i32> {
  %g = tuple.get %t, 0 : tuple<i32, i64> -> i32
  %m = tuple.make(%g : i32) : tuple<i32>
  return %m : tuple<i32>
}

// -----
// empty tuple — zero-operand make

// CHECK-LABEL: func.func @noop_empty
// CHECK: %[[E:.+]] = tuple.make : tuple<>
// CHECK: return %[[E]] : tuple<>
func.func @noop_empty() -> tuple<> {
  %e = tuple.make : tuple<>
  return %e : tuple<>
}

// -----
// get at a non-zero index, plus the trailing element survives

// CHECK-LABEL: func.func @noop_get_nonzero
// CHECK: %[[G:.+]] = tuple.get %arg0, 2 : tuple<i32, i32, i64> -> i64
// CHECK: return %[[G]] : i64
func.func @noop_get_nonzero(%t: tuple<i32, i32, i64>) -> i64 {
  %g = tuple.get %t, 2 : tuple<i32, i32, i64> -> i64
  return %g : i64
}

// -----
// nested tuple inside a make, projected back out: the get folds to the inner
// make, and the outer make, left unused, is erased

// CHECK-LABEL: func.func @noop_nested
// CHECK: %[[INNER:.+]] = tuple.make(%arg0, %arg1 : i32, i32) : tuple<i32, i32>
// CHECK-NOT: tuple.
// CHECK: return %[[INNER]] : tuple<i32, i32>
func.func @noop_nested(%a: i32, %b: i32, %c: i64) -> tuple<i32, i32> {
  %inner = tuple.make(%a, %b : i32, i32) : tuple<i32, i32>
  %outer = tuple.make(%inner, %c : tuple<i32, i32>, i64) : tuple<tuple<i32, i32>, i64>
  %g = tuple.get %outer, 0 : tuple<tuple<i32, i32>, i64> -> tuple<i32, i32>
  return %g : tuple<i32, i32>
}

// -----
// multi-element make with the same SSA value used twice

// CHECK-LABEL: func.func @noop_multi_element
// CHECK: %[[M:.+]] = tuple.make(%arg0, %arg0, %arg1 : i32, i32, i64) : tuple<i32, i32, i64>
// CHECK: return %[[M]] : tuple<i32, i32, i64>
func.func @noop_multi_element(%a: i32, %b: i64) -> tuple<i32, i32, i64> {
  %m = tuple.make(%a, %a, %b : i32, i32, i64) : tuple<i32, i32, i64>
  return %m : tuple<i32, i32, i64>
}

// -----
// a make result with multiple uses: each get folds to the operand it projects,
// nothing is duplicated, and the make, left unused, is erased

// CHECK-LABEL: func.func @noop_multi_use
// CHECK-NOT: tuple.
// CHECK: %[[S:.+]] = arith.addi %arg0, %arg1 : i32
// CHECK: return %[[S]] : i32
func.func @noop_multi_use(%a: i32, %b: i32) -> i32 {
  %m = tuple.make(%a, %b : i32, i32) : tuple<i32, i32>
  %g0 = tuple.get %m, 0 : tuple<i32, i32> -> i32
  %g1 = tuple.get %m, 1 : tuple<i32, i32> -> i32
  %s = arith.addi %g0, %g1 : i32
  return %s : i32
}
