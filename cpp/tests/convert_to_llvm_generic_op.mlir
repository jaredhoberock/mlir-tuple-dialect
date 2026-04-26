// RUN: mlir-opt --allow-unregistered-dialect --pass-pipeline="builtin.module(convert-tuple-to-llvm)" %s | FileCheck %s

// Tuple conversion must update tuple-typed operands and results on foreign
// operations. The tuple dialect should not need a pattern for every dialect
// that can temporarily carry a tuple-typed value.

// CHECK-LABEL: func.func @foreign_op_with_tuple_types
// CHECK-SAME: (%[[A:.*]]: i64, %[[B:.*]]: i64) -> i64
func.func @foreign_op_with_tuple_types(%a: i64, %b: i64) -> i64 {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i64, i64)>
  // CHECK: %[[T0:.*]] = llvm.insertvalue %[[A]], %[[UNDEF]][0] : !llvm.struct<(i64, i64)>
  // CHECK: %[[T1:.*]] = llvm.insertvalue %[[B]], %[[T0]][1] : !llvm.struct<(i64, i64)>
  %tuple = tuple.make(%a, %b : i64, i64) : tuple<i64, i64>
  // CHECK: %[[FOREIGN:.*]] = "foreign.identity"(%[[T1]]) : (!llvm.struct<(i64, i64)>) -> !llvm.struct<(i64, i64)>
  %foreign = "foreign.identity"(%tuple)
      : (tuple<i64, i64>) -> tuple<i64, i64>
  // CHECK: %[[RESULT:.*]] = llvm.extractvalue %[[FOREIGN]][1] : !llvm.struct<(i64, i64)>
  %result = tuple.get %foreign, 1 : tuple<i64, i64> -> i64
  // CHECK: return %[[RESULT]] : i64
  return %result : i64
}
