// RUN: mlir-opt --pass-pipeline="builtin.module(tuple-canonicalize)" %s | FileCheck %s

// CHECK-LABEL: func.func @get_make
// CHECK-SAME: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64) -> i64
// CHECK-NEXT: return %[[ARG1]] : i64
func.func @get_make(%arg0: i64, %arg1: i64) -> i64 {
  %tuple = tuple.make(%arg0, %arg1 : i64, i64) : tuple<i64, i64>
  %result = tuple.get %tuple, 1 : tuple<i64, i64> -> i64
  return %result : i64
}

// CHECK-LABEL: func.func @make_gets
// CHECK-SAME: (%[[ARG:.*]]: tuple<i64, i64>) -> tuple<i64, i64>
// CHECK-NEXT: return %[[ARG]] : tuple<i64, i64>
func.func @make_gets(%arg0: tuple<i64, i64>) -> tuple<i64, i64> {
  %x = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
  %y = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
  %tuple = tuple.make(%x, %y : i64, i64) : tuple<i64, i64>
  return %tuple : tuple<i64, i64>
}
