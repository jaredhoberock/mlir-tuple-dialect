// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(tuple-elaborate)" | FileCheck %s

// -----
// empty tuple

// CHECK-LABEL: func @fold_empty_tuple
// CHECK-NOT: tuple.foldl
// CHECK: return %arg0 : i32
func.func @fold_empty_tuple(%init: i32, %tup: tuple<>) -> i32 {
  %res = tuple.foldl %init, %tup : i32, tuple<> -> i32 {
  ^bb0(%acc: i32, %e: i32):
    yield %acc : i32
  }
  return %res : i32
}

// -----
// mixed tuple — body yields %e each iteration, so only the final get survives DCE

!A = !trait.poly<0>
!E = !trait.poly<1>

// CHECK-LABEL: func @fold_mixed_tuple
// CHECK-NOT: tuple.foldl
// CHECK: %[[G3:.+]] = tuple.get %arg1, 3 : tuple<i32, tuple<>, i64, tuple<f64>> -> tuple<f64>
// CHECK: return %[[G3]] : tuple<f64>
func.func @fold_mixed_tuple(%init: i32, %tup: tuple<i32,tuple<>,i64,tuple<f64>>) -> tuple<f64> {
  %res = tuple.foldl %init, %tup : i32, tuple<i32,tuple<>,i64,tuple<f64>> -> tuple<f64> {
  ^bb0(%acc: !A, %e: !E):
    yield %e : !E
  }
  return %res : tuple<f64>
}

// -----
// two input tuples — body discards %acc, so only the final iteration's make survives

// CHECK-LABEL: func @fold_two_input_tuples
// CHECK-NOT: tuple.foldl
// CHECK: %[[A1:.+]] = tuple.get %arg1, 1 : tuple<i32, i32> -> i32
// CHECK: %[[B1:.+]] = tuple.get %arg2, 1 : tuple<i32, i32> -> i32
// CHECK: %[[M:.+]] = tuple.make(%[[A1]], %[[B1]] : i32, i32) : tuple<i32, i32>
// CHECK: return %[[M]] : tuple<i32, i32>
func.func @fold_two_input_tuples(%init: tuple<i32,i32>, %a: tuple<i32,i32>, %b: tuple<i32,i32>) -> tuple<i32,i32> {
  %res = tuple.foldl %init, %a, %b : tuple<i32,i32>, tuple<i32,i32>, tuple<i32,i32> -> tuple<i32,i32> {
  ^bb0(%acc: tuple<i32,i32>, %e_a: i32, %e_b: i32):
    %pair = tuple.make(%e_a, %e_b : i32, i32) : tuple<i32,i32>
    yield %pair : tuple<i32,i32>
  }
  return %res : tuple<i32,i32>
}
