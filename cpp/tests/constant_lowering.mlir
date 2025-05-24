// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,convert-to-llvm)" %s | FileCheck %s

// -----

// CHECK-LABEL: llvm.func @empty_constant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @empty_constant() -> tuple<> {
  %c = tuple.constant : tuple<>
  return %c : tuple<>
}

// -----

// CHECK-LABEL: llvm.func @single_constant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @single_constant(%arg0: i64) -> tuple<i64> {
  %c = tuple.constant(%arg0 : i64) : tuple<i64>
  return %c : tuple<i64>
}

// -----

// CHECK-LABEL: llvm.func @pair_constant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @pair_constant(%arg0: i32, %arg1: i32) -> tuple<i32,i32> {
  %c = tuple.constant(%arg0, %arg1 : i32, i32) : tuple<i32,i32>
  return %c : tuple<i32,i32>
}

// -----

// CHECK-LABEL: func @nested_constant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @nested_constant(%a: i64, %b: tuple<i64,i64>) -> tuple<i64,tuple<i64,i64>> {
  %c = tuple.constant(%a, %b : i64, tuple<i64,i64>) : tuple<i64,tuple<i64,i64>>
  return %c : tuple<i64,tuple<i64,i64>>
}

// -----

// CHECK-LABEL: func @nested_middle_constant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @nested_middle_constant(%arg0: i64, %arg1: tuple<i64,i64>, %arg2: i64) -> tuple<i64,tuple<i64,i64>,i64> {
  %c = tuple.constant(%arg0, %arg1, %arg2 : i64, tuple<i64,i64>, i64) : tuple<i64,tuple<i64,i64>,i64>
  return %c : tuple<i64,tuple<i64,i64>,i64>
}

// -----

// CHECK-LABEL: func @deeply_nested_constant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @deeply_nested_constant(%arg0: tuple<i64,i64>, %arg1: tuple<i64>) -> tuple<tuple<i64,i64>,tuple<i64>> {
  %c = tuple.constant(%arg0, %arg1 : tuple<i64,i64>, tuple<i64>) : tuple<tuple<i64,i64>,tuple<i64>>
  return %c : tuple<tuple<i64,i64>,tuple<i64>>
}

// -----

// CHECK-LABEL: func @mixed_constant
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @mixed_constant(
  %arg0: tuple<>,
  %arg1: i64,
  %arg2: tuple<i64>
) -> tuple<tuple<>, i64, tuple<i64>> {
  %c = tuple.constant(%arg0, %arg1, %arg2 : tuple<>, i64, tuple<i64>) : tuple<tuple<>, i64, tuple<i64>>
  return %c : tuple<tuple<>, i64, tuple<i64>>
}
