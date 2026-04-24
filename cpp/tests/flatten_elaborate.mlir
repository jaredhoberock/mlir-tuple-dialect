// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(tuple-elaborate)" | FileCheck %s

// -----
// single inner tuple: tuple<tuple<i64,i64>> -> tuple<i64,i64>
// CHECK-LABEL: func @flatten_single_pair
// CHECK-NOT: tuple.flatten
// CHECK: %[[OUTER:.+]] = tuple.get %arg0, 0 : tuple<tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %[[I0:.+]] = tuple.get %[[OUTER]], 0 : tuple<i64, i64> -> i64
// CHECK: %[[I1:.+]] = tuple.get %[[OUTER]], 1 : tuple<i64, i64> -> i64
// CHECK: %[[M:.+]] = tuple.make(%[[I0]], %[[I1]] : i64, i64) : tuple<i64, i64>
// CHECK: return %[[M]] : tuple<i64, i64>
func.func @flatten_single_pair(%arg0: tuple<tuple<i64,i64>>) -> tuple<i64,i64> {
  %res = tuple.flatten %arg0
    : tuple<tuple<i64,i64>>
      -> tuple<i64,i64>
  return %res : tuple<i64,i64>
}

// -----
// pair of inner tuples: tuple<tuple<i32,i32>, tuple<i32,i32>>
// flattened to 4 elements
// CHECK-LABEL: func @flatten_pair_of_pairs
// CHECK-NOT: tuple.flatten
// CHECK: %[[O0:.+]] = tuple.get %arg0, 0 : tuple<tuple<i32, i32>, tuple<i32, i32>> -> tuple<i32, i32>
// CHECK: %[[E0:.+]] = tuple.get %[[O0]], 0 : tuple<i32, i32> -> i32
// CHECK: %[[E1:.+]] = tuple.get %[[O0]], 1 : tuple<i32, i32> -> i32
// CHECK: %[[O1:.+]] = tuple.get %arg0, 1 : tuple<tuple<i32, i32>, tuple<i32, i32>> -> tuple<i32, i32>
// CHECK: %[[E2:.+]] = tuple.get %[[O1]], 0 : tuple<i32, i32> -> i32
// CHECK: %[[E3:.+]] = tuple.get %[[O1]], 1 : tuple<i32, i32> -> i32
// CHECK: %[[M:.+]] = tuple.make(%[[E0]], %[[E1]], %[[E2]], %[[E3]] : i32, i32, i32, i32) : tuple<i32, i32, i32, i32>
// CHECK: return %[[M]] : tuple<i32, i32, i32, i32>
func.func @flatten_pair_of_pairs(%arg0: tuple<tuple<i32,i32>,tuple<i32,i32>>)
    -> tuple<i32,i32,i32,i32> {
  %res = tuple.flatten %arg0
    : tuple<tuple<i32,i32>,tuple<i32,i32>>
      -> tuple<i32,i32,i32,i32>
  return %res : tuple<i32,i32,i32,i32>
}

// -----
// flatten to empty: tuple<tuple<>,tuple<>> -> tuple<>
// CHECK-LABEL: func @flatten_to_empty
// CHECK-NOT: tuple.flatten
// CHECK: %[[M:.+]] = tuple.make : tuple<>
// CHECK: return %[[M]] : tuple<>
func.func @flatten_to_empty(%arg0: tuple<tuple<>,tuple<>>) -> tuple<> {
  %res = tuple.flatten %arg0
    : tuple<tuple<>,tuple<>>
      -> tuple<>
  return %res : tuple<>
}

// -----
// mixed inner shapes: tuple<tuple<i32>, tuple<i64,i64>, tuple<f32>>
// flattened to 4 elements — confirms arity summing across asymmetric inners
// CHECK-LABEL: func @flatten_mixed
// CHECK-NOT: tuple.flatten
// CHECK: %[[O0:.+]] = tuple.get %arg0, 0 : tuple<tuple<i32>, tuple<i64, i64>, tuple<f32>> -> tuple<i32>
// CHECK: %[[E0:.+]] = tuple.get %[[O0]], 0 : tuple<i32> -> i32
// CHECK: %[[O1:.+]] = tuple.get %arg0, 1 : tuple<tuple<i32>, tuple<i64, i64>, tuple<f32>> -> tuple<i64, i64>
// CHECK: %[[E1:.+]] = tuple.get %[[O1]], 0 : tuple<i64, i64> -> i64
// CHECK: %[[E2:.+]] = tuple.get %[[O1]], 1 : tuple<i64, i64> -> i64
// CHECK: %[[O2:.+]] = tuple.get %arg0, 2 : tuple<tuple<i32>, tuple<i64, i64>, tuple<f32>> -> tuple<f32>
// CHECK: %[[E3:.+]] = tuple.get %[[O2]], 0 : tuple<f32> -> f32
// CHECK: %[[M:.+]] = tuple.make(%[[E0]], %[[E1]], %[[E2]], %[[E3]] : i32, i64, i64, f32) : tuple<i32, i64, i64, f32>
// CHECK: return %[[M]] : tuple<i32, i64, i64, f32>
func.func @flatten_mixed(%arg0: tuple<tuple<i32>,tuple<i64,i64>,tuple<f32>>)
    -> tuple<i32,i64,i64,f32> {
  %res = tuple.flatten %arg0
    : tuple<tuple<i32>,tuple<i64,i64>,tuple<f32>>
      -> tuple<i32,i64,i64,f32>
  return %res : tuple<i32,i64,i64,f32>
}
