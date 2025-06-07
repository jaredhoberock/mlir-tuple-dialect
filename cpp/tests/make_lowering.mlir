// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,convert-to-llvm)" %s | FileCheck %s

// -----

// CHECK-LABEL: llvm.func @make_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_empty() -> tuple<> {
  %c = tuple.make : tuple<>
  return %c : tuple<>
}

// -----

// CHECK-LABEL: llvm.func @make_single
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_single(%arg0: i64) -> tuple<i64> {
  %c = tuple.make(%arg0 : i64) : tuple<i64>
  return %c : tuple<i64>
}

// -----

// CHECK-LABEL: llvm.func @make_pair
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_pair(%arg0: i32, %arg1: f32) -> tuple<i32,f32> {
  %c = tuple.make(%arg0, %arg1 : i32, f32) : tuple<i32,f32>
  return %c : tuple<i32,f32>
}

// -----

// CHECK-LABEL: func @make_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_nested(%a: i64, %b: tuple<i64,f64>) -> tuple<i64,tuple<i64,f64>> {
  %c = tuple.make(%a, %b : i64, tuple<i64,f64>) : tuple<i64,tuple<i64,f64>>
  return %c : tuple<i64,tuple<i64,f64>>
}

// -----

// CHECK-LABEL: func @make_nested_middle
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_nested_middle(%arg0: i64, %arg1: tuple<f64,f64>, %arg2: i64) -> tuple<i64,tuple<f64,f64>,i64> {
  %c = tuple.make(%arg0, %arg1, %arg2 : i64, tuple<f64,f64>, i64) : tuple<i64,tuple<f64,f64>,i64>
  return %c : tuple<i64,tuple<f64,f64>,i64>
}

// -----

// CHECK-LABEL: func @make_deeply_nested
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_deeply_nested(%arg0: tuple<i64,i64>, %arg1: tuple<i64>) -> tuple<tuple<i64,i64>,tuple<i64>> {
  %c = tuple.make(%arg0, %arg1 : tuple<i64,i64>, tuple<i64>) : tuple<tuple<i64,i64>,tuple<i64>>
  return %c : tuple<tuple<i64,i64>,tuple<i64>>
}

// -----

// CHECK-LABEL: func @make_mixed
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @make_mixed(
  %arg0: tuple<>,
  %arg1: i64,
  %arg2: tuple<i64>
) -> tuple<tuple<>, i64, tuple<i64>> {
  %c = tuple.make(%arg0, %arg1, %arg2 : tuple<>, i64, tuple<i64>) : tuple<tuple<>, i64, tuple<i64>>
  return %c : tuple<tuple<>, i64, tuple<i64>>
}
