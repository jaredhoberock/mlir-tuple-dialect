// RUN: mlir-opt %s | FileCheck %s

// ---- Test 1: append i64 ----

// CHECK-LABEL: func @append_i64
// CHECK: %[[C:.+]] = tuple.append %arg0, %arg1 : tuple<i64>, i64 -> tuple<i64, i64>
func.func @append_i64(%arg0: tuple<i64>, %arg1: i64) -> tuple<i64,i64> {
  %res = tuple.append %arg0, %arg1 : tuple<i64>, i64 -> tuple<i64,i64>
  return %res : tuple<i64,i64>
}

// ---- Test 2: append to empty tuple ----

// CHECK-LABEL: func @append_empty
// CHECK: %[[C:.+]] = tuple.append %arg0, %arg1 : tuple<>, i64 -> tuple<i64>
func.func @append_empty(%arg0 : tuple<>, %arg1: i64) -> tuple<i64> {
  %res = tuple.append %arg0, %arg1 : tuple<>, i64 -> tuple<i64>
  return %res : tuple<i64>
}

// ---- Test 3: append single ----

// CHECK-LABEL: func @append_single
// CHECK: %[[C:.+]] = tuple.append %arg0, %arg1 : tuple<i64>, tuple<i64> -> tuple<i64, tuple<i64>>
func.func @append_single(%arg0: tuple<i64>, %arg1: tuple<i64>) -> tuple<i64, tuple<i64>> {
  %res = tuple.append %arg0, %arg1 : tuple<i64>, tuple<i64> -> tuple<i64, tuple<i64>>
  return %res : tuple<i64, tuple<i64>>
}

// ---- Test 4: append empty to pair -----

// CHECK-LABEL: func @append_empty_to_pair
// CHECK: %[[C:.+]] = tuple.append %arg0, %arg1 : tuple<i64, i64>, tuple<> -> tuple<i64, i64, tuple<>>
func.func @append_empty_to_pair(%arg0: tuple<i64, i64>, %arg1: tuple<>) -> tuple<i64, i64, tuple<>> {
  %res = tuple.append %arg0, %arg1 : tuple<i64, i64>, tuple<> -> tuple<i64, i64, tuple<>>
  return %res : tuple<i64, i64, tuple<>>
}
