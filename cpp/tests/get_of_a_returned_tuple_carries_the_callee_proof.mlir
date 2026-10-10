// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -pass-pipeline='builtin.module(monomorphize-trait,convert-to-llvm)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s

// @make_one and @make_two each return, in a tuple, @Need at a projection
// spelling derived from @Need_from over the @Bound[i64] evidence the body
// chose: @One (7) and @Two (9) respectively. A call's result repeats its
// callee's signature, which repeats what the body returns, and a tuple.get's
// result is its element's type, so each extracted claim carries its body's
// proof and the method called through it runs the chosen impl's.

// CHECK: {{^}}79{{$}}

!S = !trait.poly<0>
!P = !trait.proj<@Trait[i64], "Output">
trait.trait private @Trait(%self: !trait.claim<@Trait[!S]>) {
  trait.assoc_type @Output
}
trait.impl private @Trait_impl(%self: !trait.claim<@Trait[i64]>) {
  trait.assoc_type @Output = i64
}
trait.trait private @Bound(%self: !trait.claim<@Bound[!S]>) { trait.method @value() -> i64 }
trait.impl private @One(%self: !trait.claim<@Bound[i64]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 7 : i64
    trait.return %v : i64
  }
}
trait.impl private @Two(%self: !trait.claim<@Bound[i64]>) {
  trait.method @value() -> i64 {
    %v = arith.constant 9 : i64
    trait.return %v : i64
  }
}
trait.trait private @Need(%self: !trait.claim<@Need[!S]>) { trait.method @get() -> i64 }
trait.impl private @Need_from(%self: !trait.claim<@Need[!S]>, %b: !trait.claim<@Bound[!S]>) {
  trait.method @get() -> i64 {
    %v = trait.method.call %b @Bound[!S]::@value() : () -> i64
    trait.return %v : i64
  }
}
func.func private @make_one() -> tuple<!trait.claim<@Need[!P]>> {
  %b = trait.witness @One for @Bound[i64]
  %eq = trait.witness proj_resolve !P resolves i64 by @Trait_impl : !trait.claim<!P = i64>
  %c = trait.coerce %b : !trait.claim<@Bound[i64] by @One> to !trait.claim<@Bound[!P]> via (%eq) : (!trait.claim<!P = i64>)
  %d = trait.derive @Need[!P] from @Need_from[!P] given(%c) : (!trait.claim<@Bound[!P]>)
  %t = tuple.make(%d : !trait.claim<@Need[!P]>) : tuple<!trait.claim<@Need[!P]>>
  return %t : tuple<!trait.claim<@Need[!P]>>
}
func.func private @make_two() -> tuple<!trait.claim<@Need[!P]>> {
  %b = trait.witness @Two for @Bound[i64]
  %eq = trait.witness proj_resolve !P resolves i64 by @Trait_impl : !trait.claim<!P = i64>
  %c = trait.coerce %b : !trait.claim<@Bound[i64] by @Two> to !trait.claim<@Bound[!P]> via (%eq) : (!trait.claim<!P = i64>)
  %d = trait.derive @Need[!P] from @Need_from[!P] given(%c) : (!trait.claim<@Bound[!P]>)
  %t = tuple.make(%d : !trait.claim<@Need[!P]>) : tuple<!trait.claim<@Need[!P]>>
  return %t : tuple<!trait.claim<@Need[!P]>>
}
func.func @main() -> i64 {
  %ones = func.call @make_one() : () -> tuple<!trait.claim<@Need[!P]>>
  %one = tuple.get %ones, 0 : tuple<!trait.claim<@Need[!P]>> -> !trait.claim<@Need[!P]>
  %twos = func.call @make_two() : () -> tuple<!trait.claim<@Need[!P]>>
  %two = tuple.get %twos, 0 : tuple<!trait.claim<@Need[!P]>> -> !trait.claim<@Need[!P]>
  %a = trait.method.call %one @Need[!P]::@get() : () -> i64
  %b = trait.method.call %two @Need[!P]::@get() : () -> i64
  %ten = arith.constant 10 : i64
  %tens = arith.muli %a, %ten : i64
  %sum = arith.addi %tens, %b : i64
  return %sum : i64
}
