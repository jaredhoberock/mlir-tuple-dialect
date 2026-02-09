// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,inline)" %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @cmp_eq_empty
// CHECK: %true = arith.constant true
// CHECK: return %true : i1
func.func @cmp_eq_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp eq, %a, %b : tuple<>, tuple<>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ne_empty
// CHECK: %false = arith.constant false
// CHECK: return %false : i1
func.func @cmp_ne_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp ne, %a, %b : tuple<>, tuple<>
  return %res : i1
}

!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait @PartialEq[!S,!O] {
  func.func private @eq(!S, !O) -> i1

  func.func @ne(%self: !S, %other: !O) -> i1 {
    %a = trait.assume @PartialEq[!S,!O]
    %equal = trait.method.call %a @PartialEq[!S,!O]::@eq(%self, %other) : (!S, !O) -> i1
    %true = arith.constant 1 : i1
    %result = arith.xori %equal, %true : i1
    return %result : i1
  }
}

trait.impl for @PartialEq[i32,i32] {
  func.func private @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    return %res : i1
  }
}

// -----
// CHECK-LABEL: func.func @cmp_eq_single_i32
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i32> -> i32
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i32> -> i32
// CHECK: %2 = arith.cmpi eq, %0, %1 : i32
// CHECK: return %2 : i1
func.func @cmp_eq_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp eq, %a, %b : tuple<i32>, tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ne_single_i32
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i32> -> i32
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i32> -> i32
// CHECK: %2 = arith.cmpi ne, %0, %1 : i32
// CHECK: return %2 : i1
func.func @cmp_ne_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp ne, %a, %b : tuple<i32>, tuple<i32>
  return %res : i1
}

trait.impl for @PartialEq[i64,i64] {
  func.func private @eq(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi eq, %self, %other : i64
    return %res : i1
  }
}

// -----
// CHECK-LABEL: func.func @cmp_eq_pair_i64
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, i64> -> i64
// CHECK: %2 = arith.cmpi eq, %0, %1 : i64
// CHECK: %3 = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
// CHECK: %4 = tuple.get %arg1, 1 : tuple<i64, i64> -> i64
// CHECK: %5 = arith.cmpi eq, %3, %4 : i64
// CHECK: %6 = arith.andi %2, %5 : i1
// CHECK: return %6 : i1
func.func @cmp_eq_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp eq, %a, %b : tuple<i64,i64>, tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ne_pair_i64
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, i64> -> i64
// CHECK: %2 = arith.cmpi ne, %0, %1 : i64
// CHECK: %3 = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
// CHECK: %4 = tuple.get %arg1, 1 : tuple<i64, i64> -> i64
// CHECK: %5 = arith.cmpi ne, %3, %4 : i64
// CHECK: %6 = arith.ori %2, %5 : i1
// CHECK: return %6 : i1
func.func @cmp_ne_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp ne, %a, %b : tuple<i64,i64>, tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_eq_nested_i64
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %2 = arith.cmpi eq, %0, %1 : i64
// CHECK: %3 = tuple.get %arg0, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %4 = tuple.get %arg1, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %5 = tuple.get %3, 0 : tuple<i64, i64> -> i64
// CHECK: %6 = tuple.get %4, 0 : tuple<i64, i64> -> i64
// CHECK: %7 = arith.cmpi eq, %5, %6 : i64
// CHECK: %8 = tuple.get %3, 1 : tuple<i64, i64> -> i64
// CHECK: %9 = tuple.get %4, 1 : tuple<i64, i64> -> i64
// CHECK: %10 = arith.cmpi eq, %8, %9 : i64
// CHECK: %11 = arith.andi %7, %10 : i1
// CHECK: %12 = arith.andi %2, %11 : i1
// CHECK: return %12 : i1
func.func @cmp_eq_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp eq, %a, %b : tuple<i64,tuple<i64,i64>>, tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ne_nested_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %2 = arith.cmpi ne, %0, %1 : i64
// CHECK: %3 = tuple.get %arg0, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %4 = tuple.get %arg1, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %5 = tuple.get %3, 0 : tuple<i64, i64> -> i64
// CHECK: %6 = tuple.get %4, 0 : tuple<i64, i64> -> i64
// CHECK: %7 = arith.cmpi eq, %5, %6 : i64
// CHECK: %8 = tuple.get %3, 1 : tuple<i64, i64> -> i64
// CHECK: %9 = tuple.get %4, 1 : tuple<i64, i64> -> i64
// CHECK: %10 = arith.cmpi eq, %8, %9 : i64
// CHECK: %11 = arith.andi %7, %10 : i1
// CHECK: %12 = arith.xori %11, %true : i1
// CHECK: %13 = arith.ori %2, %12 : i1
// CHECK: return %13 : i1
func.func @cmp_ne_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp ne, %a, %b : tuple<i64,tuple<i64,i64>>, tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_eq_mixed_i64
// CHECK: %0 = tuple.get %arg0, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %2 = arith.cmpi eq, %0, %1 : i64
// CHECK: %3 = tuple.get %arg0, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %4 = tuple.get %arg1, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %5 = tuple.get %3, 0 : tuple<i64> -> i64
// CHECK: %6 = tuple.get %4, 0 : tuple<i64> -> i64
// CHECK: %7 = arith.cmpi eq, %5, %6 : i64
// CHECK: %8 = arith.andi %2, %7 : i1
// CHECK: return %8 : i1
!MixedTuple = tuple<tuple<>, i64, tuple<i64>>
func.func @cmp_eq_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp eq, %a, %b : !MixedTuple, !MixedTuple
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ne_mixed_i64
// CHECK: %0 = tuple.get %arg0, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %2 = arith.cmpi ne, %0, %1 : i64
// CHECK: %3 = tuple.get %arg0, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %4 = tuple.get %arg1, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %5 = tuple.get %3, 0 : tuple<i64> -> i64
// CHECK: %6 = tuple.get %4, 0 : tuple<i64> -> i64
// CHECK: %7 = arith.cmpi ne, %5, %6 : i64
// CHECK: %8 = arith.ori %2, %7 : i1
// CHECK: return %8 : i1
func.func @cmp_ne_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp ne, %a, %b : !MixedTuple, !MixedTuple
  return %res : i1
}
