// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,canonicalize,inline)" %s | FileCheck %s

// -----
// CHECK-LABEL: func.func @cmp_lt_empty
// CHECK: %false = arith.constant false
// CHECK: return %false : i1
func.func @cmp_lt_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp lt, %a, %b : tuple<>, tuple<>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_le_empty
// CHECK: %true = arith.constant true
// CHECK: return %true : i1
func.func @cmp_le_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp le, %a, %b : tuple<>, tuple<>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_gt_empty
// CHECK: %false = arith.constant false
// CHECK: return %false : i1
func.func @cmp_gt_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp gt, %a, %b : tuple<>, tuple<>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ge_empty
// CHECK: %true = arith.constant true
// CHECK: return %true : i1
func.func @cmp_ge_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp ge, %a, %b : tuple<>, tuple<>
  return %res : i1
}

!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait @PartialOrd[!S,!O] {
  func.func private @lt(!S, !O) -> i1
  func.func private @le(!S, !O) -> i1
  func.func private @gt(!S, !O) -> i1
  func.func private @ge(!S, !O) -> i1
}

trait.impl for @PartialOrd[i32,i32] {
  func.func private @lt(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi slt, %self, %other : i32
    return %res : i1
  }
  func.func private @le(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi sle, %self, %other : i32
    return %res : i1
  }
  func.func private @gt(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi sgt, %self, %other : i32
    return %res : i1
  }
  func.func private @ge(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi sge, %self, %other : i32
    return %res : i1
  }
}

// -----
// CHECK-LABEL: func.func @cmp_lt_single_i32
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i32> -> i32
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i32> -> i32
// CHECK: %2 = arith.cmpi slt, %0, %1 : i32
// CHECK: return %2 : i1
func.func @cmp_lt_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp lt, %a, %b : tuple<i32>, tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_le_single_i32
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i32> -> i32
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i32> -> i32
// CHECK: %2 = arith.cmpi slt, %0, %1 : i32
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i32
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = arith.ori %2, %5 : i1
// CHECK: return %6 : i1
func.func @cmp_le_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp le, %a, %b : tuple<i32>, tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_gt_single_i32
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i32> -> i32
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i32> -> i32
// CHECK: %2 = arith.cmpi sgt, %0, %1 : i32
// CHECK: return %2 : i1
func.func @cmp_gt_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp gt, %a, %b : tuple<i32>, tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ge_single_i32
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i32> -> i32
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i32> -> i32
// CHECK: %2 = arith.cmpi slt, %0, %1 : i32
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i32
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = arith.ori %3, %5 : i1
// CHECK: return %6 : i1
func.func @cmp_ge_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp ge, %a, %b : tuple<i32>, tuple<i32>
  return %res : i1
}

trait.impl for @PartialOrd[i64,i64] {
  func.func private @lt(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi slt, %self, %other : i64
    return %res : i1
  }
  func.func private @le(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi sle, %self, %other : i64
    return %res : i1
  }
  func.func private @gt(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi sgt, %self, %other : i64
    return %res : i1
  }
  func.func private @ge(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi sge, %self, %other : i64
    return %res : i1
  }
}

// -----
// CHECK-LABEL: func.func @cmp_lt_pair_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, i64> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
// CHECK: %7 = tuple.get %arg1, 1 : tuple<i64, i64> -> i64
// CHECK: %8 = arith.cmpi slt, %6, %7 : i64
// CHECK: %9 = arith.andi %5, %8 : i1
// CHECK: %10 = arith.ori %2, %9 : i1
// CHECK: return %10 : i1
func.func @cmp_lt_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp lt, %a, %b : tuple<i64,i64>, tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_le_pair_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, i64> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
// CHECK: %7 = tuple.get %arg1, 1 : tuple<i64, i64> -> i64
// CHECK: %8 = arith.cmpi slt, %6, %7 : i64
// CHECK: %9 = arith.cmpi sgt, %6, %7 : i64
// CHECK: %10 = arith.ori %8, %9 : i1
// CHECK: %11 = arith.xori %10, %true : i1
// CHECK: %12 = arith.andi %5, %8 : i1
// CHECK: %13 = arith.ori %2, %12 : i1
// CHECK: %14 = arith.andi %5, %11 : i1
// CHECK: %15 = arith.ori %13, %14 : i1
// CHECK: return %15 : i1
func.func @cmp_le_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp le, %a, %b : tuple<i64,i64>, tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_gt_pair_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, i64> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
// CHECK: %7 = tuple.get %arg1, 1 : tuple<i64, i64> -> i64
// CHECK: %8 = arith.cmpi sgt, %6, %7 : i64
// CHECK: %9 = arith.andi %5, %8 : i1
// CHECK: %10 = arith.ori %3, %9 : i1
// CHECK: return %10 : i1
func.func @cmp_gt_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp gt, %a, %b : tuple<i64,i64>, tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ge_pair_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, i64> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
// CHECK: %7 = tuple.get %arg1, 1 : tuple<i64, i64> -> i64
// CHECK: %8 = arith.cmpi slt, %6, %7 : i64
// CHECK: %9 = arith.cmpi sgt, %6, %7 : i64
// CHECK: %10 = arith.ori %8, %9 : i1
// CHECK: %11 = arith.xori %10, %true : i1
// CHECK: %12 = arith.andi %5, %9 : i1
// CHECK: %13 = arith.ori %3, %12 : i1
// CHECK: %14 = arith.andi %5, %11 : i1
// CHECK: %15 = arith.ori %13, %14 : i1
// CHECK: return %15 : i1
func.func @cmp_ge_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp ge, %a, %b : tuple<i64,i64>, tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_lt_nested_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %7 = tuple.get %arg1, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %8 = tuple.get %6, 0 : tuple<i64, i64> -> i64
// CHECK: %9 = tuple.get %7, 0 : tuple<i64, i64> -> i64
// CHECK: %10 = arith.cmpi slt, %8, %9 : i64
// CHECK: %11 = arith.cmpi sgt, %8, %9 : i64
// CHECK: %12 = arith.ori %10, %11 : i1
// CHECK: %13 = arith.xori %12, %true : i1
// CHECK: %14 = tuple.get %6, 1 : tuple<i64, i64> -> i64
// CHECK: %15 = tuple.get %7, 1 : tuple<i64, i64> -> i64
// CHECK: %16 = arith.cmpi slt, %14, %15 : i64
// CHECK: %17 = arith.andi %13, %16 : i1
// CHECK: %18 = arith.ori %10, %17 : i1
// CHECK: %19 = arith.andi %5, %18 : i1
// CHECK: %20 = arith.ori %2, %19 : i1
// CHECK: return %20 : i1
func.func @cmp_lt_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp lt, %a, %b : tuple<i64,tuple<i64,i64>>, tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_le_nested_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %7 = tuple.get %arg1, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %8 = tuple.get %6, 0 : tuple<i64, i64> -> i64
// CHECK: %9 = tuple.get %7, 0 : tuple<i64, i64> -> i64
// CHECK: %10 = arith.cmpi slt, %8, %9 : i64
// CHECK: %11 = arith.cmpi sgt, %8, %9 : i64
// CHECK: %12 = arith.ori %10, %11 : i1
// CHECK: %13 = arith.xori %12, %true : i1
// CHECK: %14 = tuple.get %6, 1 : tuple<i64, i64> -> i64
// CHECK: %15 = tuple.get %7, 1 : tuple<i64, i64> -> i64
// CHECK: %16 = arith.cmpi slt, %14, %15 : i64
// CHECK: %17 = arith.andi %13, %16 : i1
// CHECK: %18 = arith.ori %10, %17 : i1
// CHECK: %19 = tuple.get %6, 0 : tuple<i64, i64> -> i64
// CHECK: %20 = tuple.get %7, 0 : tuple<i64, i64> -> i64
// CHECK: %21 = arith.cmpi slt, %19, %20 : i64
// CHECK: %22 = arith.cmpi sgt, %19, %20 : i64
// CHECK: %23 = arith.ori %21, %22 : i1
// CHECK: %24 = arith.xori %23, %true : i1
// CHECK: %25 = tuple.get %6, 1 : tuple<i64, i64> -> i64
// CHECK: %26 = tuple.get %7, 1 : tuple<i64, i64> -> i64
// CHECK: %27 = arith.cmpi sgt, %25, %26 : i64
// CHECK: %28 = arith.andi %24, %27 : i1
// CHECK: %29 = arith.ori %22, %28 : i1
// CHECK: %30 = arith.ori %18, %29 : i1
// CHECK: %31 = arith.xori %30, %true : i1
// CHECK: %32 = arith.andi %5, %18 : i1
// CHECK: %33 = arith.ori %2, %32 : i1
// CHECK: %34 = arith.andi %5, %31 : i1
// CHECK: %35 = arith.ori %33, %34 : i1
// CHECK: return %35 : i1
func.func @cmp_le_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp le, %a, %b : tuple<i64,tuple<i64,i64>>, tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_gt_nested_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %7 = tuple.get %arg1, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %8 = tuple.get %6, 0 : tuple<i64, i64> -> i64
// CHECK: %9 = tuple.get %7, 0 : tuple<i64, i64> -> i64
// CHECK: %10 = arith.cmpi slt, %8, %9 : i64
// CHECK: %11 = arith.cmpi sgt, %8, %9 : i64
// CHECK: %12 = arith.ori %10, %11 : i1
// CHECK: %13 = arith.xori %12, %true : i1
// CHECK: %14 = tuple.get %6, 1 : tuple<i64, i64> -> i64
// CHECK: %15 = tuple.get %7, 1 : tuple<i64, i64> -> i64
// CHECK: %16 = arith.cmpi sgt, %14, %15 : i64
// CHECK: %17 = arith.andi %13, %16 : i1
// CHECK: %18 = arith.ori %11, %17 : i1
// CHECK: %19 = arith.andi %5, %18 : i1
// CHECK: %20 = arith.ori %3, %19 : i1
// CHECK: return %20 : i1
func.func @cmp_gt_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp gt, %a, %b : tuple<i64,tuple<i64,i64>>, tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ge_nested_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 0 : tuple<i64, tuple<i64, i64>> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %7 = tuple.get %arg1, 1 : tuple<i64, tuple<i64, i64>> -> tuple<i64, i64>
// CHECK: %8 = tuple.get %6, 0 : tuple<i64, i64> -> i64
// CHECK: %9 = tuple.get %7, 0 : tuple<i64, i64> -> i64
// CHECK: %10 = arith.cmpi slt, %8, %9 : i64
// CHECK: %11 = arith.cmpi sgt, %8, %9 : i64
// CHECK: %12 = arith.ori %10, %11 : i1
// CHECK: %13 = arith.xori %12, %true : i1
// CHECK: %14 = tuple.get %6, 1 : tuple<i64, i64> -> i64
// CHECK: %15 = tuple.get %7, 1 : tuple<i64, i64> -> i64
// CHECK: %16 = arith.cmpi slt, %14, %15 : i64
// CHECK: %17 = arith.andi %13, %16 : i1
// CHECK: %18 = arith.ori %10, %17 : i1
// CHECK: %19 = tuple.get %6, 0 : tuple<i64, i64> -> i64
// CHECK: %20 = tuple.get %7, 0 : tuple<i64, i64> -> i64
// CHECK: %21 = arith.cmpi slt, %19, %20 : i64
// CHECK: %22 = arith.cmpi sgt, %19, %20 : i64
// CHECK: %23 = arith.ori %21, %22 : i1
// CHECK: %24 = arith.xori %23, %true : i1
// CHECK: %25 = tuple.get %6, 1 : tuple<i64, i64> -> i64
// CHECK: %26 = tuple.get %7, 1 : tuple<i64, i64> -> i64
// CHECK: %27 = arith.cmpi sgt, %25, %26 : i64
// CHECK: %28 = arith.andi %24, %27 : i1
// CHECK: %29 = arith.ori %22, %28 : i1
// CHECK: %30 = arith.ori %18, %29 : i1
// CHECK: %31 = arith.xori %30, %true : i1
// CHECK: %32 = arith.andi %5, %29 : i1
// CHECK: %33 = arith.ori %3, %32 : i1
// CHECK: %34 = arith.andi %5, %31 : i1
// CHECK: %35 = arith.ori %33, %34 : i1
// CHECK: return %35 : i1
func.func @cmp_ge_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp ge, %a, %b : tuple<i64,tuple<i64,i64>>, tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_lt_mixed_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %7 = tuple.get %arg1, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %8 = tuple.get %6, 0 : tuple<i64> -> i64
// CHECK: %9 = tuple.get %7, 0 : tuple<i64> -> i64
// CHECK: %10 = arith.cmpi slt, %8, %9 : i64
// CHECK: %11 = arith.andi %5, %10 : i1
// CHECK: %12 = arith.ori %2, %11 : i1
// CHECK: return %12 : i1
!MixedTuple = tuple<tuple<>, i64, tuple<i64>>
func.func @cmp_lt_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp lt, %a, %b : !MixedTuple, !MixedTuple
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_le_mixed_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %7 = tuple.get %arg1, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %8 = tuple.get %6, 0 : tuple<i64> -> i64
// CHECK: %9 = tuple.get %7, 0 : tuple<i64> -> i64
// CHECK: %10 = arith.cmpi slt, %8, %9 : i64
// CHECK: %11 = tuple.get %6, 0 : tuple<i64> -> i64
// CHECK: %12 = tuple.get %7, 0 : tuple<i64> -> i64
// CHECK: %13 = arith.cmpi sgt, %11, %12 : i64
// CHECK: %14 = arith.ori %10, %13 : i1
// CHECK: %15 = arith.xori %14, %true : i1
// CHECK: %16 = arith.andi %5, %10 : i1
// CHECK: %17 = arith.ori %2, %16 : i1
// CHECK: %18 = arith.andi %5, %15 : i1
// CHECK: %19 = arith.ori %17, %18 : i1
// CHECK: return %19 : i1
func.func @cmp_le_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp le, %a, %b : !MixedTuple, !MixedTuple
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_gt_mixed_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %7 = tuple.get %arg1, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %8 = tuple.get %6, 0 : tuple<i64> -> i64
// CHECK: %9 = tuple.get %7, 0 : tuple<i64> -> i64
// CHECK: %10 = arith.cmpi sgt, %8, %9 : i64
// CHECK: %11 = arith.andi %5, %10 : i1
// CHECK: %12 = arith.ori %3, %11 : i1
// CHECK: return %12 : i1
func.func @cmp_gt_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp gt, %a, %b : !MixedTuple, !MixedTuple
  return %res : i1
}

// -----
// CHECK-LABEL: func.func @cmp_ge_mixed_i64
// CHECK: %true = arith.constant true
// CHECK: %0 = tuple.get %arg0, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %1 = tuple.get %arg1, 1 : tuple<tuple<>, i64, tuple<i64>> -> i64
// CHECK: %2 = arith.cmpi slt, %0, %1 : i64
// CHECK: %3 = arith.cmpi sgt, %0, %1 : i64
// CHECK: %4 = arith.ori %2, %3 : i1
// CHECK: %5 = arith.xori %4, %true : i1
// CHECK: %6 = tuple.get %arg0, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %7 = tuple.get %arg1, 2 : tuple<tuple<>, i64, tuple<i64>> -> tuple<i64>
// CHECK: %8 = tuple.get %6, 0 : tuple<i64> -> i64
// CHECK: %9 = tuple.get %7, 0 : tuple<i64> -> i64
// CHECK: %10 = arith.cmpi slt, %8, %9 : i64
// CHECK: %11 = tuple.get %6, 0 : tuple<i64> -> i64
// CHECK: %12 = tuple.get %7, 0 : tuple<i64> -> i64
// CHECK: %13 = arith.cmpi sgt, %11, %12 : i64
// CHECK: %14 = arith.ori %10, %13 : i1
// CHECK: %15 = arith.xori %14, %true : i1
// CHECK: %16 = arith.andi %5, %13 : i1
// CHECK: %17 = arith.ori %3, %16 : i1
// CHECK: %18 = arith.andi %5, %15 : i1
// CHECK: %19 = arith.ori %17, %18 : i1
// CHECK: return %19 : i1
func.func @cmp_ge_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp ge, %a, %b : !MixedTuple, !MixedTuple
  return %res : i1
}
