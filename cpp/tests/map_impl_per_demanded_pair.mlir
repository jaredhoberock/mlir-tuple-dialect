// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(resolve-impls-trait)" -verify-diagnostics | FileCheck %s

// The map generator answers the pair of tuple types it was asked about: two
// demands of different arities get one impl each, whose assumptions and
// @Claims binding are read off the element types of those very tuples. A
// demand whose two tuples disagree on arity has no elementwise reading at all,
// so it gets no impl.

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

trait.impl private for @PartialEq[f64,f64] {
  func.func @eq(%self: f64, %other: f64) -> i1 {
    %res = arith.cmpf oeq, %self, %other : f64
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

func.func @claims_of_a_singleton()
    -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims"> {
  %c = trait.allege @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
  %claims = trait.method.call %c @tuple.MapPartialEq[tuple<i32>, tuple<i32>]::@claims()
    : () -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
  return %claims : !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
}

func.func @claims_of_a_pair()
    -> !trait.proj<@tuple.MapPartialEq[tuple<i32,f64>, tuple<i32,f64>], "Claims"> {
  %c = trait.allege @tuple.MapPartialEq[tuple<i32,f64>, tuple<i32,f64>]
  %claims = trait.method.call %c @tuple.MapPartialEq[tuple<i32,f64>, tuple<i32,f64>]::@claims()
    : () -> !trait.proj<@tuple.MapPartialEq[tuple<i32,f64>, tuple<i32,f64>], "Claims">
  return %claims : !trait.proj<@tuple.MapPartialEq[tuple<i32,f64>, tuple<i32,f64>], "Claims">
}

// the pair's impl assumes @PartialEq at each of its two element positions
// CHECK: trait.impl private for @tuple.MapPartialEq[tuple<i32, f64>, tuple<i32, f64>]
// CHECK-SAME: where [@PartialEq[i32, i32], @PartialEq[f64, f64]]
// CHECK-NEXT: trait.assoc_type @Claims = tuple<!trait.claim<@PartialEq[i32, i32]>, !trait.claim<@PartialEq[f64, f64]>>

// the singleton's impl is its own, with the one assumption its one position needs
// CHECK: trait.impl private for @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
// CHECK-SAME: where [@PartialEq[i32, i32]]
// CHECK-NEXT: trait.assoc_type @Claims = tuple<!trait.claim<@PartialEq[i32, i32]>>

// -----

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

func.func @claims_of_tuples_of_unequal_arity()
    -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32,f64>], "Claims"> {
  // expected-error @+2 {{no impl with satisfiable assumptions}}
  // expected-error @+1 {{unresolved monomorphic trait.allege}}
  %c = trait.allege @tuple.MapPartialEq[tuple<i32>, tuple<i32,f64>]
  %claims = trait.method.call %c @tuple.MapPartialEq[tuple<i32>, tuple<i32,f64>]::@claims()
    : () -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32,f64>], "Claims">
  return %claims : !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32,f64>], "Claims">
}
