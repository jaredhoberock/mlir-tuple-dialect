// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(llvm-target-to-data-layout,monomorphize-trait,convert-to-llvm)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The equality generator serves the trait its tag names, here @Eq. The impl it
// generates compares under @Eq, the trait its tuple.cmp names, so the
// elementwise claims, the mapper and the fold all read @Eq's @eq: comparing a
// one-element tuple with itself prints 1.

// CHECK: 1

!S = !trait.poly<0>
!O = !trait.poly<1>
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  trait.trait private @Eq(!trait.claim<@Eq[!S, !O]>) members [@Eq.eq] attributes {tuple.impl_generator = "partial_eq"}
  func.func private @Eq.eq(!trait.claim<@Eq[!S, !O]>, !S, !O) -> i1
  trait.impl private @Eq_i64(%s: !trait.claim<@Eq[i64, i64]>) members [@Eq.eq -> @Eq_i64.eq] {
  }
  func.func private @Eq_i64.eq(%s: !trait.claim<@Eq[i64, i64]>, %a: i64, %b: i64) -> i1 {
    %v = arith.cmpi eq, %a, %b : i64
    return %v : i1
  }
  func.func @main() -> i64 {
    %n = arith.constant 7 : i64
    %t = tuple.make(%n : i64) : tuple<i64>
    %a = trait.allege @Eq[tuple<i64>, tuple<i64>]
    %same = trait.func.call @Eq.eq(%a, %t, %t) : (!trait.claim<@Eq[tuple<i64>, tuple<i64>]>, tuple<i64>, tuple<i64>) -> i1
    %r = arith.extui %same : i1 to i64
    return %r : i64
  }
}
