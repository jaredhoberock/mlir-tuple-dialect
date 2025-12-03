// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(monomorphize-trait)" | FileCheck %s

// -----
// empty input tuple: result is just (init)
// CHECK-LABEL: func @exclusive_scan_empty
// CHECK: tuple.make
// CHECK-NOT: tuple.exclusive_scan
// CHECK-NOT: tuple.foldl
func.func @exclusive_scan_empty(%arg0: tuple<>, %init: i64) -> tuple<i64> {
  %res = tuple.exclusive_scan %arg0, %init : tuple<>, i64 -> tuple<i64> {
  ^bb0(%acc: i64, %elem: i64):
    %next = arith.muli %acc, %elem : i64
    yield %next : i64
  }
  return %res : tuple<i64>
}

// -----
// singleton i64 -> (init, init*e0)
// CHECK-LABEL: func @exclusive_scan_singleton
// CHECK: tuple.get %arg0, 0
// CHECK: arith.muli
// CHECK: tuple.make
// CHECK-NOT: tuple.exclusive_scan
// CHECK-NOT: tuple.foldl
func.func @exclusive_scan_singleton(%arg0: tuple<i64>, %init: i64) -> tuple<i64, i64> {
  %res = tuple.exclusive_scan %arg0, %init : tuple<i64>, i64 -> tuple<i64, i64> {
  ^bb0(%acc: i64, %elem: i64):
    %next = arith.muli %acc, %elem : i64
    yield %next : i64
  }
  return %res : tuple<i64, i64>
}

// -----
// triple with multiply: (init, init*e0, init*e0*e1, init*e0*e1*e2)
// CHECK-LABEL: func @exclusive_scan_triple_multiply
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.get %arg0, 2
// CHECK: arith.muli
// CHECK: tuple.make
// CHECK-NOT: tuple.exclusive_scan
// CHECK-NOT: tuple.foldl
func.func @exclusive_scan_triple_multiply(%arg0: tuple<i64, i64, i64>, %init: i64)
    -> tuple<i64, i64, i64, i64> {
  %res = tuple.exclusive_scan %arg0, %init
      : tuple<i64, i64, i64>, i64 -> tuple<i64, i64, i64, i64> {
  ^bb0(%acc: i64, %elem: i64):
    %next = arith.muli %acc, %elem : i64
    yield %next : i64
  }
  return %res : tuple<i64, i64, i64, i64>
}

// -----
// pair with add: (init, init+e0, init+e0+e1)
// CHECK-LABEL: func @exclusive_scan_add_pair
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: arith.addi
// CHECK: tuple.make
// CHECK-NOT: tuple.exclusive_scan
// CHECK-NOT: tuple.foldl
func.func @exclusive_scan_add_pair(%arg0: tuple<i32, i32>, %init: i32)
    -> tuple<i32, i32, i32> {
  %res = tuple.exclusive_scan %arg0, %init
      : tuple<i32, i32>, i32 -> tuple<i32, i32, i32> {
  ^bb0(%acc: i32, %elem: i32):
    %next = arith.addi %acc, %elem : i32
    yield %next : i32
  }
  return %res : tuple<i32, i32, i32>
}

// -----
// polymorphic element type (unused), monomorphic tuple input
// CHECK-LABEL: func @exclusive_scan_poly_elem
// We only care that exclusive_scan lowers and uses the prefix state, not %arg0.
//
// CHECK: %0 = tuple.make(%arg1 : i64) : tuple<i64>
// CHECK: %1 = tuple.get %0, 0 : tuple<i64> -> i64
// CHECK: %4 = tuple.make(%{{.*}}, %{{.*}} : i64, i64) : tuple<i64, i64>
// CHECK: %5 = tuple.get %4, 1 : tuple<i64, i64> -> i64
!E = !trait.poly<0>
func.func @exclusive_scan_poly_elem(%arg0: tuple<i32, f32>, %arg1: i64)
    -> tuple<i64, i64, i64> {
  %result = tuple.exclusive_scan %arg0, %arg1
      : tuple<i32, f32>, i64 -> tuple<i64, i64, i64> {
  ^bb0(%acc: i64, %elem: !E):
    %c1_i64 = arith.constant 1 : i64
    %next = arith.addi %acc, %c1_i64 : i64
    yield %next : i64
  }
  return %result : tuple<i64, i64, i64>
}
