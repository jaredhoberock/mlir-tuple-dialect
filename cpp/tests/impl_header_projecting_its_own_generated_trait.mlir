// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-to-llvm)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @I's header projects through its own trait, @H, at an application the
// homogeneous-tuple generator serves. Reading @I's header for @H[i64] asks
// for @H[tuple<i64>], whose candidates include @I's header again; there it is
// no candidate, the generator mints @H[tuple<i64>]'s impl binding Element =
// i64, and @I serves @H[i64].

// CHECK: 7

!T = !trait.poly<0>
trait.trait private @H(%s: !trait.claim<@H[!T]>) attributes {tuple.impl_generator = "homogeneous_tuple"} {
  trait.assoc_type @Element
  trait.method @v() -> i64 { %v = arith.constant 7 : i64 trait.return %v : i64 }
}
trait.impl private @I(%s: !trait.claim<@H[!trait.proj<@H[tuple<i64>], "Element">]>) { trait.assoc_type @Element = i32 }
func.func @main() -> i64 {
  %a = trait.allege @H[i64]
  %v = trait.method.call %a @H[i64]::@v() : () -> i64
  return %v : i64
}
