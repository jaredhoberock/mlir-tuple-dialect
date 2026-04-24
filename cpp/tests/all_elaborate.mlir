// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(tuple-elaborate)" | FileCheck %s

// -----
// empty tuple — vacuously true

// CHECK-LABEL: func @all_empty_tuple
// CHECK-NOT: tuple.all
// CHECK-NOT: tuple.foldl
// CHECK: %[[T:.+]] = arith.constant true
// CHECK: return %[[T]] : i1
func.func @all_empty_tuple(%tup: tuple<>) -> i1 {
  %res = tuple.all %tup : tuple<> {
  ^bb0(%e: !trait.poly<0>):
    // body won't run; any predicate is fine
    %false = arith.constant false
    tuple.yield %false : i1
  }
  return %res : i1
}

// -----
// tuple<i1, i1> with identity predicate → folds with AND; after DCE of the
// vacuous init `true`, a single `arith.andi` of the two elements remains.

// CHECK-LABEL: func @all_i1_identity
// CHECK-NOT: tuple.all
// CHECK-NOT: tuple.foldl
// CHECK-NOT: tuple.yield
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<i1, i1> -> i1
// CHECK: %[[G1:.+]] = tuple.get %arg0, 1 : tuple<i1, i1> -> i1
// CHECK: %[[R:.+]] = arith.andi %[[G0]], %[[G1]] : i1
// CHECK: return %[[R]] : i1
func.func @all_i1_identity(%tup: tuple<i1, i1>) -> i1 {
  %res = tuple.all %tup : tuple<i1, i1> {
  ^bb0(%e: i1):
    tuple.yield %e : i1
  }
  return %res : i1
}
