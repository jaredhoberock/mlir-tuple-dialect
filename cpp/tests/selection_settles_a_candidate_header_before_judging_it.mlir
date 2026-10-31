// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: split-file %s %t
// RUN: mlir-opt %t/spelled_first.mlir -pass-pipeline='builtin.module(llvm-target-to-data-layout,monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s
// RUN: mlir-opt %t/demanded_first.mlir -pass-pipeline='builtin.module(llvm-target-to-data-layout,monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s
// RUN: mlir-opt %t/demanded_twice.mlir -pass-pipeline='builtin.module(llvm-target-to-data-layout,monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s
// RUN: mlir-opt %t/settled_by_another_demand.mlir -pass-pipeline='builtin.module(llvm-target-to-data-layout,monomorphize-trait,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' | mlir-runner -e main --entry-point-result=i64 | FileCheck %s --check-prefix=SUM

// @I serves @A at the element type of a one-element tuple, a projection only
// the homogeneous-tuple generator settles. Selecting for @A[i64] reads @I's
// header through selection, which generates the impl binding that element
// type before @I is judged, so @I serves @A[i64] wherever the demand stands:
// after a function spelling the projection, before it, demanded twice, or
// after another demand settled the projection first. A candidate judged with
// its header still unsettled would be refused, and the refusal kept.

// CHECK: 7
// SUM: 17

//--- spelled_first.mlir
!T = !trait.poly<0>
!E = !trait.proj<@H[tuple<i64>], "Element">
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  trait.trait private @H(!trait.claim<@H[!T]>) assoc {Element = 0} attributes {tuple.impl_generator = "homogeneous_tuple"}
  trait.trait private @A(!trait.claim<@A[!T]>) members [@A.value]
  func.func private @A.value(!trait.claim<@A[!T]>) -> i64
  trait.impl private @I(%s: !trait.claim<@A[!E]>) members [@A.value -> @I.value] {
  }
  func.func private @I.value(%s: !trait.claim<@A[!E]>) -> i64 {
    %c = arith.constant 7 : i64 return %c : i64
  }
  func.func @head(%x: !E) -> !E { return %x : !E }
  func.func @main() -> i64 {
    %a = trait.allege @A[i64]
    %v = trait.func.call @A.value(%a) : (!trait.claim<@A[i64]>) -> i64
    return %v : i64
  }
}

//--- demanded_first.mlir
!T = !trait.poly<0>
!E = !trait.proj<@H[tuple<i64>], "Element">
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  trait.trait private @H(!trait.claim<@H[!T]>) assoc {Element = 0} attributes {tuple.impl_generator = "homogeneous_tuple"}
  trait.trait private @A(!trait.claim<@A[!T]>) members [@A.value]
  func.func private @A.value(!trait.claim<@A[!T]>) -> i64
  trait.impl private @I(%s: !trait.claim<@A[!E]>) members [@A.value -> @I.value] {
  }
  func.func private @I.value(%s: !trait.claim<@A[!E]>) -> i64 {
    %c = arith.constant 7 : i64 return %c : i64
  }
  func.func @main() -> i64 {
    %a = trait.allege @A[i64]
    %v = trait.func.call @A.value(%a) : (!trait.claim<@A[i64]>) -> i64
    return %v : i64
  }
  func.func @head(%x: !E) -> !E { return %x : !E }
}

//--- demanded_twice.mlir
!T = !trait.poly<0>
!E = !trait.proj<@H[tuple<i64>], "Element">
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  trait.trait private @H(!trait.claim<@H[!T]>) assoc {Element = 0} attributes {tuple.impl_generator = "homogeneous_tuple"}
  trait.trait private @A(!trait.claim<@A[!T]>) members [@A.value]
  func.func private @A.value(!trait.claim<@A[!T]>) -> i64
  trait.impl private @I(%s: !trait.claim<@A[!E]>) members [@A.value -> @I.value] {
  }
  func.func private @I.value(%s: !trait.claim<@A[!E]>) -> i64 {
    %c = arith.constant 7 : i64 return %c : i64
  }
  func.func @repeat() -> i64 { %a = trait.allege @A[i64] %n = arith.constant 2 : i64 return %n : i64 }
  func.func @head(%x: !E) -> !E { return %x : !E }
  func.func @main() -> i64 {
    %a = trait.allege @A[i64]
    %v = trait.func.call @A.value(%a) : (!trait.claim<@A[i64]>) -> i64
    return %v : i64
  }
}

//--- settled_by_another_demand.mlir
!T = !trait.poly<0>
module attributes {llvm.target = #llvm.target<triple = "x86_64-unknown-linux-gnu", chip = "">} {
  trait.trait private @G(!trait.claim<@G[!T]>) assoc {Element = 0} members [default @G.value] attributes {tuple.impl_generator = "homogeneous_tuple"}
  func.func private @G.value(%s: !trait.claim<@G[!T]>) -> i64 {
    %c = arith.constant 0 : i64 return %c : i64
  }
  trait.trait private @A(!trait.claim<@A[!T]>) members [@A.value]
  func.func private @A.value(!trait.claim<@A[!T]>) -> i64
  trait.impl private @I(%s: !trait.claim<@A[!trait.proj<@G[tuple<i32>], "Element">]>) members [@A.value -> @I.value] {
  }
  func.func private @I.value(%s: !trait.claim<@A[!trait.proj<@G[tuple<i32>], "Element">]>) -> i64 {
    %c = arith.constant 17 : i64 return %c : i64
  }
  func.func @main() -> i64 {
    %g = trait.allege @G[tuple<i32>]
    %vg = trait.func.call @G.value(%g) : (!trait.claim<@G[tuple<i32>]>) -> i64
    %a = trait.allege @A[i32]
    %va = trait.func.call @A.value(%a) : (!trait.claim<@A[i32]>) -> i64
    %r = arith.addi %vg, %va : i64
    return %r : i64
  }
}
