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

// CHECK-LABEL: func.func @mixed_signature
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i64, i64)>) -> !trait.poly<0>
func.func @mixed_signature(%tuple: tuple<i64, i64>) -> !trait.poly<0> {
  %result = ub.poison : !trait.poly<0>
  // CHECK: return %{{.*}} : !trait.poly<0>
  return %result : !trait.poly<0>
}

// CHECK-LABEL: func.func @return_memref_from_mixed_signature
// CHECK-SAME: -> memref<?x?xi64>
func.func @return_memref_from_mixed_signature(
    %tuple: tuple<i64, i64>
) -> memref<?x?xi64> {
  %memref = ub.poison : memref<?x?xi64>
  // CHECK: return %{{.*}} : memref<?x?xi64>
  return %memref : memref<?x?xi64>
}

// CHECK-LABEL: func.func @return_index_from_mixed_signature
// CHECK-SAME: (%{{.*}}: i8) -> index
func.func @return_index_from_mixed_signature(%unit: tuple<>) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: return %{{.*}} : index
  return %c0 : index
}
