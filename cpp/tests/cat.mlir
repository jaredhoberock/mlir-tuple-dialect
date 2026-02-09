// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: single-element + single-element ----

// CHECK-LABEL: func @cat_single_single
// CHECK: %[[LHS:.+]] = tuple.make(%arg0 : i64) : tuple<i64>
// CHECK: %[[RHS:.+]] = tuple.make(%arg1 : i64) : tuple<i64>
// CHECK: %[[CAT:.+]] = tuple.cat %[[LHS]], %[[RHS]] : tuple<i64>, tuple<i64> -> tuple<i64, i64>
// CHECK: return %[[CAT]] : tuple<i64, i64>
func.func @cat_single_single(%arg0: i64, %arg1: i64) -> tuple<i64, i64> {
  %lhs = tuple.make(%arg0 : i64) : tuple<i64>
  %rhs = tuple.make(%arg1 : i64) : tuple<i64>
  %r = tuple.cat %lhs, %rhs : tuple<i64>, tuple<i64> -> tuple<i64, i64>
  return %r : tuple<i64, i64>
}

// ---- Test 2: empty + tuple ----

// CHECK-LABEL: func @cat_empty_lhs
// CHECK: %[[E:.+]] = tuple.make : tuple<>
// CHECK: %[[RHS:.+]] = tuple.make(%arg0 : i64) : tuple<i64>
// CHECK: %[[CAT:.+]] = tuple.cat %[[E]], %[[RHS]] : tuple<>, tuple<i64>
// CHECK: return %[[CAT]] : tuple<i64>
func.func @cat_empty_lhs(%arg0: i64) -> tuple<i64> {
  %e = tuple.make : tuple<>
  %rhs = tuple.make(%arg0 : i64) : tuple<i64>
  %r = tuple.cat %e, %rhs : tuple<>, tuple<i64> -> tuple<i64>
  return %r : tuple<i64>
}

// ---- Test 3: tuple + empty ----

// CHECK-LABEL: func @cat_empty_rhs
// CHECK: %[[LHS:.+]] = tuple.make(%arg0 : i64) : tuple<i64>
// CHECK: %[[E:.+]] = tuple.make : tuple<>
// CHECK: %[[CAT:.+]] = tuple.cat %[[LHS]], %[[E]] : tuple<i64>, tuple<>
// CHECK: return %[[CAT]] : tuple<i64>
func.func @cat_empty_rhs(%arg0: i64) -> tuple<i64> {
  %lhs = tuple.make(%arg0 : i64) : tuple<i64>
  %e = tuple.make : tuple<>
  %r = tuple.cat %lhs, %e : tuple<i64>, tuple<> -> tuple<i64>
  return %r : tuple<i64>
}

// ---- Test 4: (i64) + (i64,i64) ----

// CHECK-LABEL: func @cat_pair_and_single
// CHECK: %[[A:.+]] = tuple.make(%arg0 : i64) : tuple<i64>
// CHECK: %[[B:.+]] = tuple.make(%arg1, %arg2 : i64, i64) : tuple<i64, i64>
// CHECK: %[[CAT:.+]] = tuple.cat %[[A]], %[[B]] : tuple<i64>, tuple<i64, i64>
// CHECK: return %[[CAT]] : tuple<i64, i64, i64>
func.func @cat_pair_and_single(%arg0: i64, %arg1: i64, %arg2: i64)
      -> tuple<i64,i64,i64> {
  %a = tuple.make(%arg0 : i64) : tuple<i64>
  %b = tuple.make(%arg1, %arg2 : i64, i64) : tuple<i64, i64>
  %r = tuple.cat %a, %b : tuple<i64>, tuple<i64, i64> -> tuple<i64, i64, i64>
  return %r : tuple<i64, i64, i64>
}

// ---- Test 5: nested (tuple<i64>, tuple<i64,i64>) ----

// CHECK-LABEL: func @cat_nested
// CHECK: %[[L:.+]] = tuple.make(%arg0 : tuple<i64>) : tuple<tuple<i64>>
// CHECK: %[[R:.+]] = tuple.make(%arg1 : tuple<i64, i64>) : tuple<tuple<i64, i64>>
// CHECK: %[[CAT:.+]] = tuple.cat %[[L]], %[[R]] : tuple<tuple<i64>>, tuple<tuple<i64, i64>>
// CHECK: return %[[CAT]] : tuple<tuple<i64>, tuple<i64, i64>>
func.func @cat_nested(%arg0: tuple<i64>, %arg1: tuple<i64,i64>)
      -> tuple<tuple<i64>,tuple<i64,i64>> {
  %l = tuple.make(%arg0 : tuple<i64>) : tuple<tuple<i64>>
  %r = tuple.make(%arg1 : tuple<i64,i64>) : tuple<tuple<i64, i64>>
  %c = tuple.cat %l, %r : tuple<tuple<i64>>, tuple<tuple<i64, i64>> -> tuple<tuple<i64>, tuple<i64, i64>>
  return %c : tuple<tuple<i64>, tuple<i64, i64>>
}
 
// ---- Test 6: mixed (tuple<>, i64) + (tuple<i64>) ----

// CHECK-LABEL: func @cat_mixed
// CHECK: %[[L:.+]] = tuple.make(%arg0, %arg1 : tuple<>, i64) : tuple<tuple<>, i64>
// CHECK: %[[R:.+]] = tuple.make(%arg2 : tuple<i64>) : tuple<tuple<i64>>
// CHECK: %[[CAT:.+]] = tuple.cat %[[L]], %[[R]] : tuple<tuple<>, i64>, tuple<tuple<i64>>
// CHECK: return %[[CAT]] : tuple<tuple<>, i64, tuple<i64>>
func.func @cat_mixed(%arg0: tuple<>, %arg1: i64, %arg2: tuple<i64>)
      -> tuple<tuple<>, i64, tuple<i64>> {
  %l = tuple.make(%arg0, %arg1 : tuple<>, i64) : tuple<tuple<>, i64>
  %r = tuple.make(%arg2 : tuple<i64>) : tuple<tuple<i64>>
  %c = tuple.cat %l, %r : tuple<tuple<>, i64>, tuple<tuple<i64>> -> tuple<tuple<>, i64, tuple<i64>>
  return %c : tuple<tuple<>, i64, tuple<i64>>
}

// ---- Test 7: polymorphic + concrete ----

// CHECK-LABEL: func @cat_poly_left
// CHECK: %[[CAT:.+]] = tuple.cat %arg0, %arg1 : !tuple.poly<0>, tuple<i32> -> !tuple.poly<1>
// CHECK: return %[[CAT]] : !tuple.poly<1>
func.func @cat_poly_left(%a: !tuple.poly<0>, %b: tuple<i32>) -> !tuple.poly<1> {
  %r = tuple.cat %a, %b : !tuple.poly<0>, tuple<i32> -> !tuple.poly<1>
  return %r : !tuple.poly<1>
}

// ---- Test 8: concrete + polymorphic ----

// CHECK-LABEL: func @cat_poly_right
// CHECK: %[[CAT:.+]] = tuple.cat %arg0, %arg1 : tuple<i32>, !tuple.poly<0> -> !tuple.poly<1>
// CHECK: return %[[CAT]] : !tuple.poly<1>
func.func @cat_poly_right(%a: tuple<i32>, %b: !tuple.poly<0>) -> !tuple.poly<1> {
  %r = tuple.cat %a, %b : tuple<i32>, !tuple.poly<0> -> !tuple.poly<1>
  return %r : !tuple.poly<1>
}
