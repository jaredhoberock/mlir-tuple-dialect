// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(llvm-target-to-data-layout,monomorphize-trait,convert-to-llvm)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @I's header projects through its own trait, @H, at an application the
// homogeneous-tuple generator serves. Reading @I's header for @H[i64] asks
// for @H[tuple<i64>], whose candidates include @I's header again; there it is
// no candidate, the generator mints @H[tuple<i64>]'s impl binding Element =
// i64, and @I serves @H[i64].

// CHECK: 7

!T = !trait.poly<0>
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  trait.trait private @H(!trait.claim<@H[!T]>) assoc {Element = 0} members [@H.v] attributes {tuple.impl_generator = "homogeneous_tuple"}
  func.func private @H.v(%s: !trait.claim<@H[!T]>) -> i64 {
    %v = arith.constant 7 : i64 return %v : i64
  }
  trait.impl private @I(%s: !trait.claim<@H[!trait.proj<@H[tuple<i64>], "Element">]>) bindings {Element = i32} {
  }
  func.func @main() -> i64 {
    %a = trait.allege @H[i64]
    %v = trait.func.call @H.v(%a) : (!trait.claim<@H[i64]>) -> i64
    return %v : i64
  }
}
