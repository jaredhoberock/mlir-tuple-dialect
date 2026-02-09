// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// ---- empty tuple
// CHECK-LABEL: func @exclusive_scan_empty
// CHECK: tuple.exclusive_scan
func.func @exclusive_scan_empty(%input: tuple<>, %init: i64) -> tuple<i64> {
  %result = tuple.exclusive_scan %input, %init : tuple<>, i64 -> tuple<i64> {
  ^bb0(%acc: i64, %elem: i64):
    %next = arith.muli %acc, %elem : i64
    yield %next : i64
  }
  return %result : tuple<i64>
}

// ---- singleton tuple
// CHECK-LABEL: func @exclusive_scan_singleton
// CHECK: tuple.exclusive_scan
func.func @exclusive_scan_singleton(%input: tuple<i64>, %init: i64) -> tuple<i64, i64> {
  %result = tuple.exclusive_scan %input, %init : tuple<i64>, i64 -> tuple<i64, i64> {
  ^bb0(%acc: i64, %elem: i64):
    %next = arith.muli %acc, %elem : i64
    yield %next : i64
  }
  return %result : tuple<i64, i64>
}

// ---- triple with multiply
// CHECK-LABEL: func @exclusive_scan_multiply
// CHECK: tuple.exclusive_scan
func.func @exclusive_scan_multiply(%input: tuple<i64, i64, i64>, %init: i64) -> tuple<i64, i64, i64, i64> {
  %result = tuple.exclusive_scan %input, %init : tuple<i64, i64, i64>, i64 -> tuple<i64, i64, i64, i64> {
  ^bb0(%acc: i64, %elem: i64):
    %next = arith.muli %acc, %elem : i64
    yield %next : i64
  }
  return %result : tuple<i64, i64, i64, i64>
}

// ---- pair with add
// CHECK-LABEL: func @exclusive_scan_add
// CHECK: tuple.exclusive_scan
func.func @exclusive_scan_add(%input: tuple<i32, i32>, %init: i32) -> tuple<i32, i32, i32> {
  %result = tuple.exclusive_scan %input, %init : tuple<i32, i32>, i32 -> tuple<i32, i32, i32> {
  ^bb0(%acc: i32, %elem: i32):
    %next = arith.addi %acc, %elem : i32
    yield %next : i32
  }
  return %result : tuple<i32, i32, i32>
}

// ---- unknown arity (polymorphic input/result)
// CHECK-LABEL: func @exclusive_scan_poly
// CHECK: tuple.exclusive_scan
!T = !tuple.poly<0>
!R = !tuple.poly<1>
!E = !trait.poly<0>
func.func @exclusive_scan_poly(%input: !T, %init: i64) -> !R {
  %result = tuple.exclusive_scan %input, %init : !T, i64 -> !R {
  ^bb0(%acc: i64, %elem: !E):
    %c2 = arith.constant 2 : i64
    %next = arith.muli %acc, %c2 : i64
    yield %next : i64
  }
  return %result : !R
}
