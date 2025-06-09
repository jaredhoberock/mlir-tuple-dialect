// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// -----
// empty tuple

// CHECK-LABEL: func @fold_empty_tuple
// CHECK: return %arg0
func.func @fold_empty_tuple(%init: i32, %tup: tuple<>) -> i32 {
  %res = tuple.foldl %init, %tup : i32, tuple<> -> i32 {
  ^bb0(%acc: i32, %e: i32):
    yield %acc : i32
  }
  return %res : i32
}

// -----
// mixed tuple

!A = !trait.poly<0>
!E = !trait.poly<1>

// CHECK-LABEL: func @fold_mixed_tuple
// CHECK: tuple.get %arg1, 3
func.func @fold_mixed_tuple(%init: i32, %tup: tuple<i32,tuple<>,i64,tuple<f64>>) -> tuple<f64> {
  %res = tuple.foldl %init, %tup : i32, tuple<i32,tuple<>,i64,tuple<f64>> -> tuple<f64> {
  ^bb0(%acc: !A, %e: !E):
    yield %e : !E
  }
  return %res : tuple<f64>
}
