// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// A tuple.constant round-trips its nested value attribute and its result type.

// ---- Test 1: single i64 ----
// CHECK-LABEL: func @constant_single
// CHECK: tuple.constant([42]) : tuple<i64>
func.func @constant_single() -> tuple<i64> {
  %c = tuple.constant([42 : i64]) : tuple<i64>
  return %c : tuple<i64>
}

// ---- Test 2: pair of scalars ----
// CHECK-LABEL: func @constant_pair
// CHECK: tuple.constant([7 : i32, 1.500000e+00 : f32]) : tuple<i32, f32>
func.func @constant_pair() -> tuple<i32, f32> {
  %c = tuple.constant([7 : i32, 1.5 : f32]) : tuple<i32, f32>
  return %c : tuple<i32, f32>
}

// ---- Test 3: nested tuple mirrors the struct tree ----
// CHECK-LABEL: func @constant_nested
// CHECK: tuple.constant([42, [7 : i32, 1.500000e+00 : f32]]) : tuple<i64, tuple<i32, f32>>
func.func @constant_nested() -> tuple<i64, tuple<i32, f32>> {
  %c = tuple.constant([42 : i64, [7 : i32, 1.5 : f32]]) : tuple<i64, tuple<i32, f32>>
  return %c : tuple<i64, tuple<i32, f32>>
}

// ---- Test 4: index leaf ----
// CHECK-LABEL: func @constant_index
// CHECK: tuple.constant([5 : index]) : tuple<index>
func.func @constant_index() -> tuple<index> {
  %c = tuple.constant([5 : index]) : tuple<index>
  return %c : tuple<index>
}

// ---- Test 5: empty tuple ----
// CHECK-LABEL: func @constant_empty
// CHECK: tuple.constant([]) : tuple<>
func.func @constant_empty() -> tuple<> {
  %c = tuple.constant([]) : tuple<>
  return %c : tuple<>
}
