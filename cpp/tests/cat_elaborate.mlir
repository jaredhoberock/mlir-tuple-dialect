// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(tuple-elaborate)" %s | FileCheck %s

// -----
// Both empty -> single empty make
// CHECK-LABEL: func.func @cat_both_empty
// CHECK-NOT: tuple.cat
// CHECK: %[[M0:.+]] = tuple.make : tuple<>
// CHECK: return %[[M0]] : tuple<>
func.func @cat_both_empty(%a : tuple<>, %b : tuple<>) -> tuple<> {
  %r = tuple.cat %a, %b : tuple<>, tuple<> -> tuple<>
  return %r : tuple<>
}

// -----
// Left empty -> picks rhs elements in order
// CHECK-LABEL: func.func @cat_left_empty
// CHECK-NOT: tuple.cat
// CHECK: %[[G0:.+]] = tuple.get %arg1, 0 : tuple<i32> -> i32
// CHECK: %[[M0:.+]] = tuple.make(%[[G0]] : i32) : tuple<i32>
// CHECK: return %[[M0]] : tuple<i32>
func.func @cat_left_empty(%a : tuple<>, %b : tuple<i32>) -> tuple<i32> {
  %r = tuple.cat %a, %b : tuple<>, tuple<i32> -> tuple<i32>
  return %r : tuple<i32>
}

// -----
// Right empty -> keeps lhs elements in order
// CHECK-LABEL: func.func @cat_right_empty
// CHECK-NOT: tuple.cat
// CHECK: %[[L0:.+]] = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
// CHECK: %[[L1:.+]] = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
// CHECK: %[[M0:.+]] = tuple.make(%[[L0]], %[[L1]] : i64, i64) : tuple<i64, i64>
// CHECK: return %[[M0]] : tuple<i64, i64>
func.func @cat_right_empty(%a : tuple<i64,i64>, %b : tuple<>) -> tuple<i64,i64> {
  %r = tuple.cat %a, %b : tuple<i64,i64>, tuple<> -> tuple<i64,i64>
  return %r : tuple<i64,i64>
}

// -----
// Flat concat: (i32, i64) ++ (i8) -> (i32, i64, i8)
// CHECK-LABEL: func.func @cat_flat_2_plus_1
// CHECK-NOT: tuple.cat
// CHECK: %[[A0:.+]] = tuple.get %arg0, 0 : tuple<i32, i64> -> i32
// CHECK: %[[A1:.+]] = tuple.get %arg0, 1 : tuple<i32, i64> -> i64
// CHECK: %[[B0:.+]] = tuple.get %arg1, 0 : tuple<i8> -> i8
// CHECK: %[[M0:.+]] = tuple.make(%[[A0]], %[[A1]], %[[B0]] : i32, i64, i8) : tuple<i32, i64, i8>
// CHECK: return %[[M0]] : tuple<i32, i64, i8>
func.func @cat_flat_2_plus_1(%a : tuple<i32,i64>, %b : tuple<i8>) -> tuple<i32,i64,i8> {
  %r = tuple.cat %a, %b : tuple<i32,i64>, tuple<i8> -> tuple<i32,i64,i8>
  return %r : tuple<i32,i64,i8>
}

// -----
// Nested on the left: (i32, tuple<i64, i8>) ++ (i1) -> (i32, tuple<i64, i8>, i1)
// (Note: the nested tuple is kept as a single element; no extra decomposition.)
// CHECK-LABEL: func.func @cat_nested_left
// CHECK-NOT: tuple.cat
// CHECK: %[[X0:.+]] = tuple.get %arg0, 0 : tuple<i32, tuple<i64, i8>> -> i32
// CHECK: %[[X1:.+]] = tuple.get %arg0, 1 : tuple<i32, tuple<i64, i8>> -> tuple<i64, i8>
// CHECK: %[[Y0:.+]] = tuple.get %arg1, 0 : tuple<i1> -> i1
// CHECK: %[[M0:.+]] = tuple.make(%[[X0]], %[[X1]], %[[Y0]] : i32, tuple<i64, i8>, i1) : tuple<i32, tuple<i64, i8>, i1>
// CHECK: return %[[M0]] : tuple<i32, tuple<i64, i8>, i1>
func.func @cat_nested_left(%a : tuple<i32, tuple<i64,i8>>, %b : tuple<i1>)
      -> tuple<i32, tuple<i64,i8>, i1> {
  %r = tuple.cat %a, %b : tuple<i32,tuple<i64,i8>>, tuple<i1> -> tuple<i32,tuple<i64,i8>,i1>
  return %r : tuple<i32,tuple<i64,i8>,i1>
}

// -----
// (i32) ++ ((i64) ++ (i8)) -> (i32,i64,i8)
// CHECK-LABEL: func.func @cat_nested_right
// CHECK-NOT: tuple.cat
// get pieces for inner cat
// CHECK: %[[B0:.+]] = tuple.get %arg1, 0 : tuple<i64> -> i64
// CHECK: %[[C0:.+]] = tuple.get %arg2, 0 : tuple<i8> -> i8
// build inner tuple<i64,i8>
// CHECK: %[[BC:.+]] = tuple.make(%[[B0]], %[[C0]] : i64, i8) : tuple<i64, i8>
// get left element
// CHECK: %[[A0:.+]] = tuple.get %arg0, 0 : tuple<i32> -> i32
// now pull from the inner tuple
// CHECK: %[[B1:.+]] = tuple.get %[[BC]], 0 : tuple<i64, i8> -> i64
// CHECK: %[[C1:.+]] = tuple.get %[[BC]], 1 : tuple<i64, i8> -> i8
// final make
// CHECK: %[[OUT:.+]] = tuple.make(%[[A0]], %[[B1]], %[[C1]] : i32, i64, i8) : tuple<i32, i64, i8>
// CHECK: return %[[OUT]] : tuple<i32, i64, i8>
func.func @cat_nested_right(%a: tuple<i32>, %b: tuple<i64>, %c: tuple<i8>) -> tuple<i32, i64, i8> {
  %bc = tuple.cat %b, %c : tuple<i64>, tuple<i8> -> tuple<i64,i8>
  %r  = tuple.cat %a, %bc : tuple<i32>, tuple<i64,i8> -> tuple<i32,i64,i8>
  return %r : tuple<i32, i64, i8>
}

// -----
// (tuple<i8,i8>, i32) ++ (i64, tuple<i1>) -> (tuple<i8,i8>, i32, i64, tuple<i1>)
// CHECK-LABEL: func.func @cat_both_nested
// CHECK-NOT: tuple.cat
// CHECK: %[[L0:.+]] = tuple.get %arg0, 0 : tuple<tuple<i8, i8>, i32> -> tuple<i8, i8>
// CHECK: %[[L1:.+]] = tuple.get %arg0, 1 : tuple<tuple<i8, i8>, i32> -> i32
// CHECK: %[[R0:.+]] = tuple.get %arg1, 0 : tuple<i64, tuple<i1>> -> i64
// CHECK: %[[R1:.+]] = tuple.get %arg1, 1 : tuple<i64, tuple<i1>> -> tuple<i1>
// CHECK: %[[M0:.+]] = tuple.make(%[[L0]], %[[L1]], %[[R0]], %[[R1]] : tuple<i8, i8>, i32, i64, tuple<i1>) : tuple<tuple<i8, i8>, i32, i64, tuple<i1>>
// CHECK: return %[[M0]] : tuple<tuple<i8, i8>, i32, i64, tuple<i1>>
func.func @cat_both_nested(%a: tuple<tuple<i8,i8>, i32>,
                           %b: tuple<i64, tuple<i1>>)
    -> tuple<tuple<i8,i8>, i32, i64, tuple<i1>> {
  %r = tuple.cat %a, %b : tuple<tuple<i8,i8>, i32>, tuple<i64, tuple<i1>> -> tuple<tuple<i8,i8>, i32, i64, tuple<i1>>
  return %r : tuple<tuple<i8,i8>, i32, i64, tuple<i1>>
}

// -----
// (tuple<>, i32) ++ (tuple<>) -> (tuple<>, i32, tuple<>)
// CHECK-LABEL: func.func @cat_preserve_empty_elements
// CHECK-NOT: tuple.cat
// CHECK: %[[L0:.+]] = tuple.get %arg0, 0 : tuple<tuple<>, i32> -> tuple<>
// CHECK: %[[L1:.+]] = tuple.get %arg0, 1 : tuple<tuple<>, i32> -> i32
// CHECK: %[[R0:.+]] = tuple.get %arg1, 0 : tuple<tuple<>> -> tuple<>
// CHECK: %[[M0:.+]] = tuple.make(%[[L0]], %[[L1]], %[[R0]] : tuple<>, i32, tuple<>) : tuple<tuple<>, i32, tuple<>>
// CHECK: return %[[M0]] : tuple<tuple<>, i32, tuple<>>
func.func @cat_preserve_empty_elements(%a: tuple<tuple<>, i32>,
                                       %b: tuple<tuple<>>)
    -> tuple<tuple<>, i32, tuple<>> {
  %r = tuple.cat %a, %b : tuple<tuple<>, i32>, tuple<tuple<>> -> tuple<tuple<>, i32, tuple<>>
  return %r : tuple<tuple<>, i32, tuple<>>
}
