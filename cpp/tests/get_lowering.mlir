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
// CHECK-LABEL: llvm.func @get_pair_i32_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @get_pair_i32_i32(%a : tuple<i32,i32>) -> tuple<i32,i32> {
  %a_0 = tuple.get %a, 0 : tuple<i32,i32> -> i32
  %a_1 = tuple.get %a, 1 : tuple<i32,i32> -> i32
  %res = tuple.constant(%a_1, %a_0 : i32, i32) : tuple<i32,i32>
  return %res : tuple<i32,i32>
}

// -----
// CHECK-LABEL: llvm.func @get_mixed
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
!MixedTuple = tuple<tuple<>, i64, tuple<i64>>
!ResultTuple = tuple<tuple<i64>, tuple<>, i64>
func.func @get_mixed(%a : !MixedTuple) -> !ResultTuple {
  %a_0 = tuple.get %a, 0 : !MixedTuple -> tuple<>
  %a_1 = tuple.get %a, 1 : !MixedTuple -> i64
  %a_2 = tuple.get %a, 2 : !MixedTuple -> tuple<i64>
  %res = tuple.constant(%a_2, %a_0, %a_1 : tuple<i64>, tuple<>, i64) : !ResultTuple
  return %res : !ResultTuple
}
