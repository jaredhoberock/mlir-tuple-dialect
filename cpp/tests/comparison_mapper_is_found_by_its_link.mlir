// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(llvm-target-to-data-layout,monomorphize-trait,convert-to-llvm)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// A mapper is the trait tagged `tuple.impl_generator = "map"` whose
// `tuple.mapped_trait` names the compared trait, never a trait of some name.
// The module declares @tuple.MapPartialEq, the name a mapper of @PartialEq is
// labelled with, but untagged and unrelated; the equality generator declares
// its own mapper under a name of its own, the map generator implements it,
// and comparing a one-element tuple with itself prints 1.

// CHECK: 1

!S = !trait.poly<0>
!O = !trait.poly<1>
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  trait.trait private @PartialEq(!trait.claim<@PartialEq[!S, !O]>) members [@PartialEq.eq] attributes {tuple.impl_generator = "partial_eq"}
  func.func private @PartialEq.eq(!trait.claim<@PartialEq[!S, !O]>, !S, !O) -> i1
  trait.trait private @tuple.MapPartialEq(!trait.claim<@tuple.MapPartialEq[!S, !O]>) assoc {Claims = 0}
  trait.impl private @PartialEq_i64(%s: !trait.claim<@PartialEq[i64, i64]>) members [@PartialEq.eq -> @PartialEq_i64.eq] {
  }
  func.func private @PartialEq_i64.eq(%s: !trait.claim<@PartialEq[i64, i64]>, %a: i64, %b: i64) -> i1 {
    %v = arith.cmpi eq, %a, %b : i64
    return %v : i1
  }
  func.func @main() -> i64 {
    %n = arith.constant 7 : i64
    %t = tuple.make(%n : i64) : tuple<i64>
    %a = trait.allege @PartialEq[tuple<i64>, tuple<i64>]
    %same = trait.func.call @PartialEq.eq(%a, %t, %t) : (!trait.claim<@PartialEq[tuple<i64>, tuple<i64>]>, tuple<i64>, tuple<i64>) -> i1
    %r = arith.extui %same : i1 to i64
    return %r : i64
  }
}
