// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// -----
// Last from singleton
// CHECK-LABEL: func.func @last_singleton
// CHECK: tuple.last
func.func @last_singleton(%a : tuple<i64>) -> i64 {
  %r = tuple.last %a : tuple<i64> -> i64
  return %r : i64
}

// -----
// Last from pair
// CHECK-LABEL: func.func @last_pair
// CHECK: tuple.last
func.func @last_pair(%a : tuple<i64, i32>) -> i32 {
  %r = tuple.last %a : tuple<i64, i32> -> i32
  return %r : i32
}

// -----
// Last from triple
// CHECK-LABEL: func.func @last_triple
// CHECK: tuple.last
func.func @last_triple(%a : tuple<i32, i64, i8>) -> i8 {
  %r = tuple.last %a : tuple<i32, i64, i8> -> i8
  return %r : i8
}

// -----
// Last extracts nested tuple
// CHECK-LABEL: func.func @last_extracts_nested
// CHECK: tuple.last
func.func @last_extracts_nested(%a : tuple<i64, tuple<i32, i32>>) -> tuple<i32, i32> {
  %r = tuple.last %a : tuple<i64, tuple<i32, i32>> -> tuple<i32, i32>
  return %r : tuple<i32, i32>
}
