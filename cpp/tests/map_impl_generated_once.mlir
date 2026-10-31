// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(instantiate-monomorphs-trait)" | FileCheck %s

// An impl generated for exactly the demanded application is a declared
// candidate of that application from then on, so a second demand of the same
// pair of tuples selects it instead of asking for another one.

!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait private @PartialEq(!trait.claim<@PartialEq[!S, !O]>) members [@PartialEq.eq] attributes {tuple.impl_generator = "partial_eq"}
func.func private @PartialEq.eq(!trait.claim<@PartialEq[!S, !O]>, !S, !O) -> i1

trait.impl private @PartialEq_i32(%self_claim: !trait.claim<@PartialEq[i32, i32]>) members [@PartialEq.eq -> @PartialEq_i32.eq] {
}
func.func private @PartialEq_i32.eq(%self_claim: !trait.claim<@PartialEq[i32, i32]>, %self: i32, %other: i32) -> i1 {
  %res = arith.cmpi eq, %self, %other : i32
  return %res : i1
}

!MS = !trait.poly<2>
!MO = !trait.poly<3>
trait.trait private @tuple.MapPartialEq(!trait.claim<@tuple.MapPartialEq[!trait.poly<0>, !trait.poly<1>]>) assoc {Claims = 0} members [@tuple.MapPartialEq.claims] attributes {
  tuple.impl_generator = "map",
  tuple.mapped_trait = @PartialEq
}
func.func private @tuple.MapPartialEq.claims(!trait.claim<@tuple.MapPartialEq[!trait.poly<0>, !trait.poly<1>]>) -> !trait.proj<@tuple.MapPartialEq[!trait.poly<0>,!trait.poly<1>], "Claims">

func.func @claims_here()
    -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims"> {
  %c = trait.allege @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
  %claims = trait.func.call @tuple.MapPartialEq.claims(%c) : (!trait.claim<@tuple.MapPartialEq[tuple<i32>, tuple<i32>]>) -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
  return %claims : !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
}

func.func @claims_there()
    -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims"> {
  %c = trait.allege @tuple.MapPartialEq[tuple<i32>, tuple<i32>]
  %claims = trait.func.call @tuple.MapPartialEq.claims(%c) : (!trait.claim<@tuple.MapPartialEq[tuple<i32>, tuple<i32>]>) -> !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
  return %claims : !trait.proj<@tuple.MapPartialEq[tuple<i32>, tuple<i32>], "Claims">
}

// both demands are served by the one impl of that application
// CHECK: trait.impl private @tuple.MapPartialEq_impl_{{h[0-9a-f]+}}(%self: !trait.claim<@tuple.MapPartialEq[tuple<i32>, tuple<i32>]>
// CHECK-NOT: trait.impl private @tuple.MapPartialEq_impl_{{h[0-9a-f]+}}(%self: !trait.claim<@tuple.MapPartialEq[tuple<i32>, tuple<i32>]>
