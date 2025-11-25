// RUN: mlir-opt %s | FileCheck %s

// ---- empty outer tuple
// CHECK-LABEL: func @flatten_empty
// CHECK: tuple.flatten
func.func @flatten_empty(%arg0: tuple<>) -> tuple<> {
  %res = tuple.flatten %arg0 : tuple<> -> tuple<>
  return %res : tuple<>
}

// ---- pair of singleton tuples: ( (i64), (i64) ) -> (i64, i64)
// CHECK-LABEL: func @flatten_pair_of_singles
// CHECK: tuple.flatten
func.func @flatten_pair_of_singles(%arg0: tuple<tuple<i64>, tuple<i64>>)
    -> tuple<i64, i64> {
  %res = tuple.flatten %arg0
    : tuple<tuple<i64>, tuple<i64>> -> tuple<i64, i64>
  return %res : tuple<i64, i64>
}

// ---- mixed inner tuple sizes: ( (i32, f32), (i64) ) -> (i32, f32, i64)
// CHECK-LABEL: func @flatten_mixed
// CHECK: tuple.flatten
func.func @flatten_mixed(%arg0: tuple<tuple<i32, f32>, tuple<i64>>)
    -> tuple<i32, f32, i64> {
  %res = tuple.flatten %arg0
    : tuple<tuple<i32, f32>, tuple<i64>> -> tuple<i32, f32, i64>
  return %res : tuple<i32, f32, i64>
}

// ----
// polymorphic TupleLike: !tuple.poly<in> -> !tuple.poly<out>
// we model that flattening may change the tuple shape, so the polys differ
// CHECK-LABEL: func @flatten_poly
// CHECK: tuple.flatten
!Tin  = !tuple.poly<0>
!Tout = !tuple.poly<1>
func.func @flatten_poly(%xs: !Tin) -> !Tout {
  %res = tuple.flatten %xs : !Tin -> !Tout
  return %res : !Tout
}
