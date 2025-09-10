// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(monomorphize-trait)" | FileCheck %s

// -----
// single i64

// CHECK-LABEL: func @map_i64_to_f32
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.make
func.func @map_i64_to_f32(%arg0: tuple<i64>) -> tuple<f32> {
  %res = tuple.map %arg0 : tuple<i64> -> tuple<f32> {
  ^bb0(%x: i64):
    %res = arith.sitofp %x : i64 to f32
    yield %res : f32
  }
  return %res : tuple<f32>
}

// -----
// pair of i64

// CHECK-LABEL: func @map_i64_i64_to_f32_f32
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.make
func.func @map_i64_i64_to_f32_f32(%arg0: tuple<i64,i64>) -> tuple<f32,f32> {
  %res = tuple.map %arg0 : tuple<i64,i64> -> tuple<f32,f32> {
  ^bb0(%x: i64):
    %res = arith.sitofp %x : i64 to f32
    yield %res : f32
  }
  return %res : tuple<f32,f32>
}

// -----
// call a method

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

// CHECK-LABEL: func @map_id_i32_f32
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.make
!X = !trait.poly<1>
func.func @map_id_i32_f32(%arg0: tuple<i32,f32>) -> tuple<i32,f32> {
  %c0 = trait.allege @Id[i32]
  %c1 = trait.allege @Id[f32]
  %claims = tuple.make (%c0, %c1 : !trait.claim<@Id[i32]>, !trait.claim<@Id[f32]>)
    : tuple<!trait.claim<@Id[i32]>, !trait.claim<@Id[f32]>>

  %res = tuple.map %arg0, %claims
    : tuple<i32,f32>, tuple<!trait.claim<@Id[i32]>, !trait.claim<@Id[f32]>>
    -> tuple<i32,f32> {
  ^bb0(%x: !X, %c: !trait.claim<@Id[!X]>):
    %res = trait.method.call %c @Id[!X]::@id(%x) : (!X) -> !X
    yield %res : !X
  }
  return %res : tuple<i32,f32>
}
