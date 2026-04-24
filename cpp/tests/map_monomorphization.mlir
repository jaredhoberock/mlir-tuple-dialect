// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(monomorphize-trait)" | FileCheck %s

// -----
// call a method — exercises trait claim resolution, impl dispatch, and
// instantiation of a distinct callee per element type (i32 vs f32).

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
// CHECK-NOT: tuple.map
// CHECK-NOT: trait.method.call
// CHECK-NOT: trait.allege
// CHECK-NOT: !trait.
// CHECK: %[[E0:.+]] = tuple.get %arg0, 0 : tuple<i32, f32> -> i32
// CHECK: %[[C0:.+]] = call @{{.+}}(%[[E0]]) : (i32) -> i32
// CHECK: %[[E1:.+]] = tuple.get %arg0, 1 : tuple<i32, f32> -> f32
// CHECK: %[[C1:.+]] = call @{{.+}}(%[[E1]]) : (f32) -> f32
// CHECK: %[[M:.+]] = tuple.make(%[[C0]], %[[C1]] : i32, f32) : tuple<i32, f32>
// CHECK: return %[[M]] : tuple<i32, f32>
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
