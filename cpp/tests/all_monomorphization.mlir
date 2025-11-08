// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(monomorphize-trait)" | FileCheck %s

// -----
// empty tuple

// CHECK-LABEL: func @all_empty_tuple
// CHECK: arith.constant true
func.func @all_empty_tuple(%tup: tuple<>) -> i1 {
  %res = tuple.all %tup : tuple<> {
  ^bb0(%e: !trait.poly<0>):
    // body won't run; any predicate is fine
    %false = arith.constant false
    tuple.yield %false : i1
  }
  return %res : i1
}

// -----
// tuple<i1, i1> with identity predicate → folds with AND

// CHECK-LABEL: func @all_i1_identity
// CHECK: tuple.get %arg0, 1
// CHECK: arith.andi
func.func @all_i1_identity(%tup: tuple<i1, i1>) -> i1 {
  %res = tuple.all %tup : tuple<i1, i1> {
  ^bb0(%e: i1):
    tuple.yield %e : i1
  }
  return %res : i1
}
