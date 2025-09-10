// RUN: mlir-opt %s | FileCheck %s

// ---- empty tuple

// CHECK-LABEL: func @map_empty
// CHECK: tuple.map
func.func @map_empty(%arg0: tuple<>) -> tuple<> {
  %res = tuple.map %arg0 : tuple<> -> tuple<> {
  ^bb0(%x: i1):
    // the types of the block argument and yield don't matter
    // because we never enter the block
    yield %x : i1
  }
  return %res : tuple<>
}

// ---- single i64

// CHECK-LABEL: func @map_i64_to_f32
// CHECK: tuple.map
func.func @map_i64_to_f32(%arg0: tuple<i64>) -> tuple<f32> {
  %res = tuple.map %arg0 : tuple<i64> -> tuple<f32> {
  ^bb0(%x: i64):
    %res = arith.sitofp %x : i64 to f32
    yield %res : f32
  }
  return %res : tuple<f32>
}

// ---- pair of i64

// CHECK-LABEL: func @map_i64_i64_to_f32_f32
// CHECK: tuple.map
func.func @map_i64_i64_to_f32_f32(%arg0: tuple<i64,i64>) -> tuple<f32,f32> {
  %res = tuple.map %arg0 : tuple<i64,i64> -> tuple<f32,f32> {
  ^bb0(%x: i64):
    %res = arith.sitofp %x : i64 to f32
    yield %res : f32
  }
  return %res : tuple<f32,f32>
}

!S = !trait.poly<0>
trait.trait @Id[!S] {
  func.func nested @id(%self: !S) -> !S
}

trait.impl for @Id[i32] {
  func.func nested @id(%self: i32) -> i32 {
    return %self : i32
  }
}

trait.impl for @Id[f32] {
  func.func nested @id(%self: f32) -> f32 {
    return %self : f32
  }
}

// ---- call a method within monomorphic map

// CHECK-LABEL: func @map_id_i32_f32
// CHECK: tuple.map
!E = !trait.poly<1>
func.func @map_id_i32_f32(
  %arg0: tuple<i32,f32>,
  %arg1: tuple<!trait.claim<@Id[i32]>, !trait.claim<@Id[f32]>>
) -> tuple<i32,f32> {

  %res = tuple.map %arg0, %arg1
    : tuple<i32,f32>, tuple<!trait.claim<@Id[i32]>, !trait.claim<@Id[f32]>>
    -> tuple<i32,f32> {
  ^bb0(%x: !E, %c: !trait.claim<@Id[!E]>):
    %res = trait.method.call %c @Id[!E]::@id(%x) : (!E) -> !E
    yield %res : !E
  }

  return %res : tuple<i32,f32>
}

// ---- unknown arity (polymorphic input/result)

// CHECK-LABEL: func @map_unknown_arity_poly
// CHECK: tuple.map
!T = !tuple.poly<0>
!U = !tuple.poly<1>
func.func @map_unknown_arity_poly(%xs: !T) -> !T {
  // body must be purely polymorphic; we just forward the element
  %res = tuple.map %xs : !T -> !T {
  ^bb0(%x: !U):
    yield %x : !U
  }
  return %res : !T
}

// ---- mix: concrete + polymorphic

// CHECK-LABEL: func @map_mix_concrete_poly
// CHECK: tuple.map
!X = !trait.poly<2>
!Y = !trait.poly<3>
func.func @map_mix_concrete_poly(
  %xs: tuple<i32,f32>,
  %ys: !T
) -> tuple<i32,f32> {
  %res = tuple.map %xs, %ys
    : tuple<i32,f32>, !T
    -> tuple<i32,f32> {
  ^bb0(%x: !X, %y: !Y):
    yield %x: !X
  }
  return %res : tuple<i32,f32>
}
