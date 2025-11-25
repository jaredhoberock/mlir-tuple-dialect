// RUN: mlir-opt %s | FileCheck %s

// ---- empty tuple
// CHECK-LABEL: func @flat_map_empty
// CHECK: tuple.flat_map
func.func @flat_map_empty(%arg0: tuple<>) -> tuple<> {
  %res = tuple.flat_map %arg0 : tuple<> -> tuple<> {
  ^bb0(%x: i1):
    // body is never executed; we just need a TupleLike yield type
    %empty = tuple.make : tuple<>
    yield %empty : tuple<>
  }
  return %res : tuple<>
}

// ---- single i64 -> singleton f32 tuple
// CHECK-LABEL: func @flat_map_i64_to_f32
// CHECK: tuple.flat_map
func.func @flat_map_i64_to_f32(%arg0: tuple<i64>) -> tuple<f32> {
  %res = tuple.flat_map %arg0 : tuple<i64> -> tuple<f32> {
  ^bb0(%x: i64):
    %f = arith.sitofp %x : i64 to f32
    %t = tuple.make (%f : f32) : tuple<f32>
    yield %t : tuple<f32>
  }
  return %res : tuple<f32>
}

// ---- pair of i64 -> each to tuple<f32,f32>, concatenated
// CHECK-LABEL: func @flat_map_i64_i64_to_f32x4
// CHECK: tuple.flat_map
func.func @flat_map_i64_i64_to_f32x4(%arg0: tuple<i64,i64>) -> tuple<f32,f32,f32,f32> {
  %res = tuple.flat_map %arg0
    : tuple<i64,i64>
    -> tuple<f32,f32,f32,f32> {
  ^bb0(%x: i64):
    %f = arith.sitofp %x : i64 to f32
    %pair = tuple.make (%f, %f : f32, f32) : tuple<f32,f32>
    yield %pair : tuple<f32,f32>
  }
  return %res : tuple<f32,f32,f32,f32>
}

// ---- unknown arity (polymorphic input/result)
// CHECK-LABEL: func @flat_map_unknown_arity_poly
// CHECK: tuple.flat_map
!T = !tuple.poly<0>
!E = !trait.poly<0>
func.func @flat_map_unknown_arity_poly(%xs: !T) -> !T {
  // body must be purely polymorphic; we wrap the element in a 1-tuple
  %res = tuple.flat_map %xs : !T -> !T {
  ^bb0(%x: !E):
    %wrapped = tuple.make (%x : !E) : tuple<!E>
    yield %wrapped : tuple<!E>
  }
  return %res : !T
}
