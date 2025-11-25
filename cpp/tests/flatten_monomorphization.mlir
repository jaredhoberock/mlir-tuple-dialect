// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(monomorphize-trait)" | FileCheck %s

// -----
// single inner tuple: tuple<tuple<i64,i64>> -> tuple<i64,i64>
// CHECK-LABEL: func @flatten_single_pair
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get
// CHECK: tuple.make
func.func @flatten_single_pair(%arg0: tuple<tuple<i64,i64>>) -> tuple<i64,i64> {
  %res = tuple.flatten %arg0
    : tuple<tuple<i64,i64>>
      -> tuple<i64,i64>
  return %res : tuple<i64,i64>
}

// -----
// pair of inner tuples: tuple<tuple<i32,i32>, tuple<i32,i32>>
// flattened to 4 elements
// CHECK-LABEL: func @flatten_pair_of_pairs
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.make
func.func @flatten_pair_of_pairs(%arg0: tuple<tuple<i32,i32>,tuple<i32,i32>>)
    -> tuple<i32,i32,i32,i32> {
  %res = tuple.flatten %arg0
    : tuple<tuple<i32,i32>,tuple<i32,i32>>
      -> tuple<i32,i32,i32,i32>
  return %res : tuple<i32,i32,i32,i32>
}

// -----
// flatten to empty: tuple<tuple<>,tuple<>> -> tuple<>
// CHECK-LABEL: func @flatten_to_empty
// CHECK: tuple.make : tuple<>
func.func @flatten_to_empty(%arg0: tuple<tuple<>,tuple<>>) -> tuple<> {
  %res = tuple.flatten %arg0
    : tuple<tuple<>,tuple<>>
      -> tuple<>
  return %res : tuple<>
}

// -----
// mixed inner shapes: tuple<tuple<i32>, tuple<i64,i64>, tuple<f32>>
// flattened to 4 elements
// CHECK-LABEL: func @flatten_mixed
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.get %arg0, 2
// CHECK: tuple.make
func.func @flatten_mixed(%arg0: tuple<tuple<i32>,tuple<i64,i64>,tuple<f32>>)
    -> tuple<i32,i64,i64,f32> {
  %res = tuple.flatten %arg0
    : tuple<tuple<i32>,tuple<i64,i64>,tuple<f32>>
      -> tuple<i32,i64,i64,f32>
  return %res : tuple<i32,i64,i64,f32>
}
