// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(resolve-impls-trait)" | FileCheck %s

// An impl generated for exactly the demanded application is a declared
// candidate of that application from then on, so a second demand of the same
// pair of tuples selects it instead of asking for another one.

!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait private @PartialEq[!S,!O] {
  func.func private @eq(!S,!O) -> i1
}

trait.impl private for @PartialEq[i32,i32] {
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    return %res : i1
  }
}

!MS = !trait.poly<2>
!MO = !trait.poly<3>
trait.trait private @tuple.MapPartialEq[!MS,!MO] attributes {
  tuple.impl_generator = "map",
  tuple.mapped_trait = @PartialEq
} {
  trait.assoc_type @Claims
  func.func private @claims() -> !trait.proj<@tuple.MapPartialEq[!MS,!MO], "Claims">
}

func.func @claims_here()
    -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims"> {
  %c = trait.allege @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
  %claims = trait.method.call %c @tuple.MapPartialEq[tuple<i32>, tuple<i32>]::@claims()
    : () -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
  return %claims : !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
}

func.func @claims_there()
    -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims"> {
  %c = trait.allege @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
  %claims = trait.method.call %c @tuple.MapPartialEq[tuple<i32>, tuple<i32>]::@claims()
    : () -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
  return %claims : !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
}

// both demands are served by the one impl of that application
// CHECK: trait.impl private for @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
// CHECK-NOT: trait.impl private for @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
