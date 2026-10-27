// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-to-llvm)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @T[i32] is derived from @TI over the hand-written proof @pb, packed into a
// tuple and taken back out. A tuple.get's result is its element's type, so the
// claim it extracts carries the proof the derive wrote, and the method called
// through it runs @TI's.

// CHECK: {{^}}9{{$}}

trait.trait private @B(%s: !trait.claim<@B[!trait.poly<0>]>) {}
trait.impl private @BI(%s: !trait.claim<@B[i32]>) {}
trait.proof private @pb {
  %b = trait.derive @B[i32] from @BI given()
  trait.return %b : !trait.claim<@B[i32]>
}
trait.trait private @T(%s: !trait.claim<@T[!trait.poly<0>]>) { trait.method @m() -> i64 }
trait.impl private @TI(%s: !trait.claim<@T[i32]>, %b: !trait.claim<@B[i32]>) {
  trait.method @m() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
func.func @main() -> i64 {
  %b = trait.witness @pb for @B[i32]
  %d = trait.derive @T[i32] from @TI given(%b) : (!trait.claim<@B[i32] by @pb>)
  %t = tuple.make(%d : !trait.claim<@T[i32]>) : tuple<!trait.claim<@T[i32]>>
  %p = tuple.get %t, 0 : tuple<!trait.claim<@T[i32]>> -> !trait.claim<@T[i32]>
  %v = trait.method.call %p @m() : !trait.claim<@T[i32]>, () -> i64
  return %v : i64
}
