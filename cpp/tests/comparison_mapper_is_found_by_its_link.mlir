// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-to-llvm)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A mapper is the trait tagged `tuple.impl_generator = "map"` whose
// `tuple.mapped_trait` names the compared trait, never a trait of some name.
// The module declares @tuple.MapPartialEq, the name a mapper of @PartialEq is
// labelled with, but untagged and unrelated; the equality generator declares
// its own mapper under a name of its own, the map generator implements it,
// and comparing a one-element tuple with itself prints 1.

// CHECK: 1

!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait private @PartialEq(%s: !trait.claim<@PartialEq[!S, !O]>) attributes {tuple.impl_generator = "partial_eq"} {
  trait.method @eq(!S, !O) -> i1
}
trait.trait private @tuple.MapPartialEq(%s: !trait.claim<@tuple.MapPartialEq[!S, !O]>) {
  trait.assoc_type @Claims
}
trait.impl private @PartialEq_i64(%s: !trait.claim<@PartialEq[i64, i64]>) {
  trait.method @eq(%a: i64, %b: i64) -> i1 {
    %v = arith.cmpi eq, %a, %b : i64
    trait.return %v : i1
  }
}
func.func @main() -> i64 {
  %n = arith.constant 7 : i64
  %t = tuple.make(%n : i64) : tuple<i64>
  %a = trait.allege @PartialEq[tuple<i64>, tuple<i64>]
  %same = trait.method.call %a @PartialEq[tuple<i64>, tuple<i64>]::@eq(%t, %t) : (tuple<i64>, tuple<i64>) -> i1
  %r = arith.extui %same : i1 to i64
  return %r : i64
}
