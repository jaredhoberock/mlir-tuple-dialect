// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(tuple-elaborate)" %s | FileCheck %s

// -----
// Last from singleton
// CHECK-LABEL: func.func @last_singleton
// CHECK-NOT: tuple.last
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<i64> -> i64
// CHECK: return %[[G0]] : i64
func.func @last_singleton(%a : tuple<i64>) -> i64 {
  %r = tuple.last %a : tuple<i64> -> i64
  return %r : i64
}

// -----
// Last from pair
// CHECK-LABEL: func.func @last_pair
// CHECK-NOT: tuple.last
// CHECK: %[[G1:.+]] = tuple.get %arg0, 1 : tuple<i64, i32> -> i32
// CHECK: return %[[G1]] : i32
func.func @last_pair(%a : tuple<i64, i32>) -> i32 {
  %r = tuple.last %a : tuple<i64, i32> -> i32
  return %r : i32
}

// -----
// Last from triple
// CHECK-LABEL: func.func @last_triple
// CHECK-NOT: tuple.last
// CHECK: %[[G2:.+]] = tuple.get %arg0, 2 : tuple<i32, i64, i8> -> i8
// CHECK: return %[[G2]] : i8
func.func @last_triple(%a : tuple<i32, i64, i8>) -> i8 {
  %r = tuple.last %a : tuple<i32, i64, i8> -> i8
  return %r : i8
}

// -----
// Last extracts nested tuple
// CHECK-LABEL: func.func @last_extracts_nested
// CHECK-NOT: tuple.last
// CHECK: %[[G1:.+]] = tuple.get %arg0, 1 : tuple<i64, tuple<i32, i32>> -> tuple<i32, i32>
// CHECK: return %[[G1]] : tuple<i32, i32>
func.func @last_extracts_nested(%a : tuple<i64, tuple<i32, i32>>) -> tuple<i32, i32> {
  %r = tuple.last %a : tuple<i64, tuple<i32, i32>> -> tuple<i32, i32>
  return %r : tuple<i32, i32>
}
