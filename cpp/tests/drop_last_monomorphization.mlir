// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait)" %s | FileCheck %s

// -----
// Drop last from pair -> singleton
// CHECK-LABEL: func.func @drop_last_pair
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<i64, i32> -> i64
// CHECK: %[[M0:.+]] = tuple.make(%[[G0]] : i64) : tuple<i64>
// CHECK: return %[[M0]] : tuple<i64>
func.func @drop_last_pair(%a : tuple<i64, i32>) -> tuple<i64> {
  %r = tuple.drop_last %a : tuple<i64, i32> -> tuple<i64>
  return %r : tuple<i64>
}

// -----
// Drop last from singleton -> empty
// CHECK-LABEL: func.func @drop_last_singleton
// CHECK: %[[M0:.+]] = tuple.make : tuple<>
// CHECK: return %[[M0]] : tuple<>
func.func @drop_last_singleton(%a : tuple<i64>) -> tuple<> {
  %r = tuple.drop_last %a : tuple<i64> -> tuple<>
  return %r : tuple<>
}

// -----
// Drop last from triple -> pair
// CHECK-LABEL: func.func @drop_last_triple
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<i32, i64, i8> -> i32
// CHECK: %[[G1:.+]] = tuple.get %arg0, 1 : tuple<i32, i64, i8> -> i64
// CHECK: %[[M0:.+]] = tuple.make(%[[G0]], %[[G1]] : i32, i64) : tuple<i32, i64>
// CHECK: return %[[M0]] : tuple<i32, i64>
func.func @drop_last_triple(%a : tuple<i32, i64, i8>) -> tuple<i32, i64> {
  %r = tuple.drop_last %a : tuple<i32, i64, i8> -> tuple<i32, i64>
  return %r : tuple<i32, i64>
}

// -----
// Drop last preserves nested tuple on the left
// CHECK-LABEL: func.func @drop_last_preserves_nested
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<tuple<i64, i64>, i32> -> tuple<i64, i64>
// CHECK: %[[M0:.+]] = tuple.make(%[[G0]] : tuple<i64, i64>) : tuple<tuple<i64, i64>>
// CHECK: return %[[M0]] : tuple<tuple<i64, i64>>
func.func @drop_last_preserves_nested(%a : tuple<tuple<i64, i64>, i32>) -> tuple<tuple<i64, i64>> {
  %r = tuple.drop_last %a : tuple<tuple<i64, i64>, i32> -> tuple<tuple<i64, i64>>
  return %r : tuple<tuple<i64, i64>>
}

// -----
// Drop last removes nested tuple
// CHECK-LABEL: func.func @drop_last_removes_nested
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<i64, tuple<i32, i32>> -> i64
// CHECK: %[[M0:.+]] = tuple.make(%[[G0]] : i64) : tuple<i64>
// CHECK: return %[[M0]] : tuple<i64>
func.func @drop_last_removes_nested(%a : tuple<i64, tuple<i32, i32>>) -> tuple<i64> {
  %r = tuple.drop_last %a : tuple<i64, tuple<i32, i32>> -> tuple<i64>
  return %r : tuple<i64>
}

// -----
// Drop last with empty tuple element preserved
// CHECK-LABEL: func.func @drop_last_preserves_empty
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<tuple<>, i32> -> tuple<>
// CHECK: %[[M0:.+]] = tuple.make(%[[G0]] : tuple<>) : tuple<tuple<>>
// CHECK: return %[[M0]] : tuple<tuple<>>
func.func @drop_last_preserves_empty(%a : tuple<tuple<>, i32>) -> tuple<tuple<>> {
  %r = tuple.drop_last %a : tuple<tuple<>, i32> -> tuple<tuple<>>
  return %r : tuple<tuple<>>
}

// -----
// Drop last chained: drop twice from triple
// CHECK-LABEL: func.func @drop_last_chained
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<i32, i64, i8> -> i32
// CHECK: %[[G1:.+]] = tuple.get %arg0, 1 : tuple<i32, i64, i8> -> i64
// CHECK: %[[M0:.+]] = tuple.make(%[[G0]], %[[G1]] : i32, i64) : tuple<i32, i64>
// CHECK: %[[G2:.+]] = tuple.get %[[M0]], 0 : tuple<i32, i64> -> i32
// CHECK: %[[M1:.+]] = tuple.make(%[[G2]] : i32) : tuple<i32>
// CHECK: return %[[M1]] : tuple<i32>
func.func @drop_last_chained(%a : tuple<i32, i64, i8>) -> tuple<i32> {
  %r1 = tuple.drop_last %a : tuple<i32, i64, i8> -> tuple<i32, i64>
  %r2 = tuple.drop_last %r1 : tuple<i32, i64> -> tuple<i32>
  return %r2 : tuple<i32>
}
