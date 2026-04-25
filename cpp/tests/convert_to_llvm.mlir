// RUN: mlir-opt --pass-pipeline="builtin.module(convert-tuple-to-llvm)" %s | FileCheck %s

// CHECK-LABEL: func.func @make_get
// CHECK-SAME: (%[[A:.*]]: i64, %[[B:.*]]: i64) -> i64
func.func @make_get(%a: i64, %b: i64) -> i64 {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i64, i64)>
  // CHECK: %[[T0:.*]] = llvm.insertvalue %[[A]], %[[UNDEF]][0] : !llvm.struct<(i64, i64)>
  // CHECK: %[[T1:.*]] = llvm.insertvalue %[[B]], %[[T0]][1] : !llvm.struct<(i64, i64)>
  %tuple = tuple.make(%a, %b : i64, i64) : tuple<i64, i64>
  // CHECK: %[[R:.*]] = llvm.extractvalue %[[T1]][0] : !llvm.struct<(i64, i64)>
  %result = tuple.get %tuple, 0 : tuple<i64, i64> -> i64
  // CHECK: return %[[R]] : i64
  return %result : i64
}

// CHECK-LABEL: func.func @signature
// CHECK-SAME: (%[[ARG:.*]]: !llvm.struct<(i64, i64)>) -> !llvm.struct<(i64, i64)>
func.func @signature(%tuple: tuple<i64, i64>) -> tuple<i64, i64> {
  // CHECK: return %[[ARG]] : !llvm.struct<(i64, i64)>
  return %tuple : tuple<i64, i64>
}
