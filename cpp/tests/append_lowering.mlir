// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,convert-to-llvm)" %s | FileCheck %s

// -----

// CHECK-LABEL: llvm.func @append_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @append_i64(%arg0: tuple<i64>, %arg1: i64) -> tuple<i64,i64> {
  %res = tuple.append %arg0, %arg1 : tuple<i64>, i64 -> tuple<i64,i64>
  return %res : tuple<i64,i64>
}

// -----

// CHECK-LABEL: llvm.func @append_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @append_empty(%arg0 : tuple<>, %arg1: i64) -> tuple<i64> {
  %res = tuple.append %arg0, %arg1 : tuple<>, i64 -> tuple<i64>
  return %res : tuple<i64>
}

// -----

// CHECK-LABEL: llvm.func @append_single
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @append_single(%arg0: tuple<i64>, %arg1: tuple<i64>) -> tuple<i64, tuple<i64>> {
  %res = tuple.append %arg0, %arg1 : tuple<i64>, tuple<i64> -> tuple<i64, tuple<i64>>
  return %res : tuple<i64, tuple<i64>>
}

// -----

// CHECK-LABEL: llvm.func @append_empty_to_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @append_empty_to_pair(%arg0: tuple<i64, i64>, %arg1: tuple<>) -> tuple<i64, i64, tuple<>> {
  %res = tuple.append %arg0, %arg1 : tuple<i64, i64>, tuple<> -> tuple<i64, i64, tuple<>>
  return %res : tuple<i64, i64, tuple<>>
}
