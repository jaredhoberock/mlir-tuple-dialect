// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-to-llvm)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// The equality generator serves the trait its tag names, here @Eq. The impl it
// generates compares under @Eq, the trait its tuple.cmp names, so the
// elementwise claims, the mapper and the fold all read @Eq's @eq: comparing a
// one-element tuple with itself prints 1.

// CHECK: 1

!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait private @Eq(%s: !trait.claim<@Eq[!S, !O]>) attributes {tuple.impl_generator = "partial_eq"} {
  trait.method @eq(!S, !O) -> i1
}
trait.impl private @Eq_i64(%s: !trait.claim<@Eq[i64, i64]>) {
  trait.method @eq(%a: i64, %b: i64) -> i1 {
    %v = arith.cmpi eq, %a, %b : i64
    trait.return %v : i1
  }
}
func.func @main() -> i64 {
  %n = arith.constant 7 : i64
  %t = tuple.make(%n : i64) : tuple<i64>
  %a = trait.allege @Eq[tuple<i64>, tuple<i64>]
  %same = trait.method.call %a @Eq[tuple<i64>, tuple<i64>]::@eq(%t, %t) : (tuple<i64>, tuple<i64>) -> i1
  %r = arith.extui %same : i1 to i64
  return %r : i64
}
