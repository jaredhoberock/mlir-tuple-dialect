// RUN: mlir-opt --convert-to-llvm %s | FileCheck %s

// -----
// CHECK-LABEL: llvm.func @get_single_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @get_single_i32(%a : tuple<i32>) -> i32 {
  %res = tuple.get %a, 0 : tuple<i32> -> i32
  return %res : i32
}

// -----
// CHECK-LABEL: llvm.func @get_pair_i32_f32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @get_pair_i32_f32(%a : tuple<i32,f32>) -> tuple<f32,i32> {
  %a_0 = tuple.get %a, 0 : tuple<i32,f32> -> i32
  %a_1 = tuple.get %a, 1 : tuple<i32,f32> -> f32
  %res = tuple.make(%a_1, %a_0 : f32, i32) : tuple<f32,i32>
  return %res : tuple<f32,i32>
}

// -----
// CHECK-LABEL: llvm.func @get_mixed
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
!MixedTuple = tuple<tuple<>, i64, tuple<f64>>
!ResultTuple = tuple<tuple<f64>, tuple<>, i64>
func.func @get_mixed(%a : !MixedTuple) -> !ResultTuple {
  %a_0 = tuple.get %a, 0 : !MixedTuple -> tuple<>
  %a_1 = tuple.get %a, 1 : !MixedTuple -> i64
  %a_2 = tuple.get %a, 2 : !MixedTuple -> tuple<f64>
  %res = tuple.make(%a_2, %a_0, %a_1 : tuple<f64>, tuple<>, i64) : !ResultTuple
  return %res : !ResultTuple
}
