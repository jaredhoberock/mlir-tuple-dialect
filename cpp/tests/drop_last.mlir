// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: drop_last from pair ----
// CHECK-LABEL: func @drop_last_pair
// CHECK: %[[C:.+]] = tuple.drop_last %arg0 : tuple<i64, i64> -> tuple<i64>
func.func @drop_last_pair(%arg0: tuple<i64, i64>) -> tuple<i64> {
  %res = tuple.drop_last %arg0 : tuple<i64, i64> -> tuple<i64>
  return %res : tuple<i64>
}

// ---- Test 2: drop_last from singleton ----
// CHECK-LABEL: func @drop_last_singleton
// CHECK: %[[C:.+]] = tuple.drop_last %arg0 : tuple<i64> -> tuple<>
func.func @drop_last_singleton(%arg0: tuple<i64>) -> tuple<> {
  %res = tuple.drop_last %arg0 : tuple<i64> -> tuple<>
  return %res : tuple<>
}

// ---- Test 3: drop_last from triple ----
// CHECK-LABEL: func @drop_last_triple
// CHECK: %[[C:.+]] = tuple.drop_last %arg0 : tuple<i64, i32, i16> -> tuple<i64, i32>
func.func @drop_last_triple(%arg0: tuple<i64, i32, i16>) -> tuple<i64, i32> {
  %res = tuple.drop_last %arg0 : tuple<i64, i32, i16> -> tuple<i64, i32>
  return %res : tuple<i64, i32>
}

// ---- Test 4: drop_last nested ----
// CHECK-LABEL: func @drop_last_nested
// CHECK: %[[C:.+]] = tuple.drop_last %arg0 : tuple<i64, tuple<i32, i32>> -> tuple<i64>
func.func @drop_last_nested(%arg0: tuple<i64, tuple<i32, i32>>) -> tuple<i64> {
  %res = tuple.drop_last %arg0 : tuple<i64, tuple<i32, i32>> -> tuple<i64>
  return %res : tuple<i64>
}

// ---- Test 5: drop_last preserves nested ----
// CHECK-LABEL: func @drop_last_preserves_nested
// CHECK: %[[C:.+]] = tuple.drop_last %arg0 : tuple<tuple<i64, i64>, i32> -> tuple<tuple<i64, i64>>
func.func @drop_last_preserves_nested(%arg0: tuple<tuple<i64, i64>, i32>) -> tuple<tuple<i64, i64>> {
  %res = tuple.drop_last %arg0 : tuple<tuple<i64, i64>, i32> -> tuple<tuple<i64, i64>>
  return %res : tuple<tuple<i64, i64>>
}
