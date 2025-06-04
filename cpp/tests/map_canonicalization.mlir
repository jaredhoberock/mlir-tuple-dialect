// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// -----
// single i64

// CHECK-LABEL: func @map_i64_to_f32
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.constant
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
// CHECK: tuple.constant
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

trait.trait @Id {
  func.func nested @id(%self: !trait.self) -> !trait.self
}

trait.impl @Id for i32 {
  func.func nested @id(%self: i32) -> i32 {
    return %self : i32
  }
}

trait.impl @Id for f32 {
  func.func nested @id(%self: f32) -> f32 {
    return %self : f32
  }
}

// CHECK-LABEL: func @map_id_i32_f32
// CHECK: tuple.get %arg0, 0
// CHECK: tuple.get %arg0, 1
// CHECK: tuple.constant
!T = !trait.poly<0,[@Id]>
func.func @map_id_i32_f32(%arg0: tuple<i32,f32>) -> tuple<i32,f32> {
  %res = tuple.map %arg0 : tuple<i32,f32> -> tuple<i32,f32> {
  ^bb0(%x: !T):
    %res = trait.method.call @Id::@id<!T>(%x) : (!trait.self) -> (!trait.self) to (!T) -> !T
    yield %res : !T
  }
  return %res : tuple<i32,f32>
}
