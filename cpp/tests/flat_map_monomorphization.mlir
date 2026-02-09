// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(monomorphize-trait)" | FileCheck %s

// -----
// single i64 -> duplicate into two elements
// CHECK-LABEL: func @flat_map_i64_dup
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.make
func.func @flat_map_i64_dup(%arg0: tuple<i64>) -> tuple<i64,i64> {
  %res = tuple.flat_map %arg0 : tuple<i64> -> tuple<i64,i64> {
  ^bb0(%x: i64):
    %t = tuple.make (%x, %x : i64, i64) : tuple<i64,i64>
    yield %t : tuple<i64,i64>
  }
  return %res : tuple<i64,i64>
}

// -----
// pair of i64 -> each element mapped to a tuple<f32,f32>
// overall result is 4 elements flattened
// CHECK-LABEL: func @flat_map_i64_i64_to_four_f32
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.make
func.func @flat_map_i64_i64_to_four_f32(%arg0: tuple<i64,i64>) -> tuple<f32,f32,f32,f32> {
  %res = tuple.flat_map %arg0 : tuple<i64,i64> -> tuple<f32,f32,f32,f32> {
  ^bb0(%x: i64):
    %a = arith.sitofp %x : i64 to f32
    %b = arith.sitofp %x : i64 to f32
    %t = tuple.make (%a, %b : f32, f32) : tuple<f32,f32>
    yield %t : tuple<f32,f32>
  }
  return %res : tuple<f32,f32,f32,f32>
}

// -----
// flat_map to empty tuple
// CHECK-LABEL: func @flat_map_to_empty
// CHECK: tuple.make : tuple<>
func.func @flat_map_to_empty(%arg0: tuple<i64, i64>) -> tuple<> {
  %res = tuple.flat_map %arg0 : tuple<i64, i64> -> tuple<> {
  ^bb0(%x: i64):
    %empty = tuple.make : tuple<>
    yield %empty : tuple<>
  }
  return %res : tuple<>
}

// -----
// flat_map where body returns a singleton tuple (map-like behavior)
// CHECK-LABEL: func @flat_map_singleton
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.make
func.func @flat_map_singleton(%arg0: tuple<i32,i32>) -> tuple<i64,i64> {
  %res = tuple.flat_map %arg0 : tuple<i32,i32> -> tuple<i64,i64> {
  ^bb0(%x: i32):
    %y = arith.extsi %x : i32 to i64
    %t = tuple.make (%y : i64) : tuple<i64>
    yield %t : tuple<i64>
  }
  return %res : tuple<i64,i64>
}

// -----
// empty input tuple, body unreachable
// CHECK-LABEL: func @flat_map_empty_input
// CHECK: tuple.make : tuple<>
func.func @flat_map_empty_input(%arg0: tuple<>) -> tuple<> {
  %res = tuple.flat_map %arg0 : tuple<> -> tuple<> {
  ^bb0(%x: i1):
    %empty = tuple.make : tuple<>
    yield %empty : tuple<>
  }
  return %res : tuple<>
}

// -----
// triple i64 -> each element duplicated into two elements
// overall result is 6 elements flattened
// CHECK-LABEL: func @flat_map_triple_dup
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.get %arg0, 2
// CHECK: tuple.make
func.func @flat_map_triple_dup(%arg0: tuple<i64,i64,i64>)
    -> tuple<i64,i64,i64,i64,i64,i64> {
  %res = tuple.flat_map %arg0
    : tuple<i64,i64,i64> -> tuple<i64,i64,i64,i64,i64,i64> {
  ^bb0(%x: i64):
    %t = tuple.make (%x, %x : i64, i64) : tuple<i64,i64>
    yield %t : tuple<i64,i64>
  }
  return %res : tuple<i64,i64,i64,i64,i64,i64>
}

// -----
// polymorphic body element type, monomorphic tuple input
// CHECK-LABEL: func @flat_map_poly_id
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.make
!E = !trait.poly<0>
func.func @flat_map_poly_id(%arg0: tuple<i32,f32>) -> tuple<i32,f32> {
  %res = tuple.flat_map %arg0
    : tuple<i32,f32> -> tuple<i32,f32> {
  ^bb0(%x: !E):
    // body is polymorphic in the element type
    %t = tuple.make(%x : !E) : tuple<!E>
    yield %t : tuple<!E>
  }
  return %res : tuple<i32,f32>
}
