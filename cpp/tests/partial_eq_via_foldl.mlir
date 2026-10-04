// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,inline)" %s | FileCheck %s

// A polymorphic tuple impl of @PartialEq written against the mapper trait:
// the impl asks the mapper for the tuple of elementwise claims and hands it to
// tuple.cmp, whose lowering folds @PartialEq::@eq across the elements. The
// mapper's impl for the demanded pair of tuples is generated here.

!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait private @PartialEq(%self: !trait.claim<@PartialEq[!S, !O]>) attributes {tuple.impl_generator = "partial_eq"} {
  trait.method @eq(!S,!O) -> i1
}

trait.impl private @PartialEq_i32(%self_claim: !trait.claim<@PartialEq[i32, i32]>) {
  trait.method @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    trait.return %res : i1
  }
}

trait.impl private @PartialEq_f64(%self_claim: !trait.claim<@PartialEq[f64, f64]>) {
  trait.method @eq(%self: f64, %other: f64) -> i1 {
    %res = arith.cmpf oeq, %self, %other : f64
    trait.return %res : i1
  }
}

// this trait maps the PartialEq trait over the elements of two tuples and
// binds the tuple of their claims as its @Claims associated type
!MapPartialEqS = !trait.poly<2>
!MapPartialEqO = !trait.poly<3>
trait.trait private @tuple.MapPartialEq(%self: !trait.claim<@tuple.MapPartialEq[!MapPartialEqS, !MapPartialEqO]>) attributes {
  tuple.impl_generator = "map",
  tuple.mapped_trait = @PartialEq
} {
  trait.assoc_type @Claims
  trait.method @claims()
    -> !trait.proj<@tuple.MapPartialEq[!MapPartialEqS,!MapPartialEqO], "Claims">
}

// this is the polymorphic tuple impl of PartialEq
!TS = !tuple.poly<5>
!TO = !tuple.poly<6>
trait.impl private @tuple.PartialEq(%self_claim: !trait.claim<@PartialEq[!TS, !TO]>, %mapped: !trait.claim<@tuple.MapPartialEq[!TS,!TO]>) {
  trait.method @eq(%self: !TS, %other: !TO) -> i1 {
    // first get a tuple of elementwise @PartialEq claims
    %claims = trait.method.call %mapped @tuple.MapPartialEq[!TS,!TO]::@claims()
      : () -> !trait.proj<@tuple.MapPartialEq[!TS,!TO], "Claims">

    // fold @PartialEq::@eq over the tuples
    %res = tuple.cmp eq @PartialEq, %self, %other, %claims
      : !TS, !TO, !trait.proj<@tuple.MapPartialEq[!TS,!TO], "Claims">
    trait.return %res : i1
  }
}

func.func @foo(%a: tuple<i32,f64>, %b: tuple<i32,f64>) -> i1 {
  %c = trait.allege @PartialEq[tuple<i32,f64>,tuple<i32,f64>]
  %res = trait.method.call %c @PartialEq[tuple<i32,f64>,tuple<i32,f64>]::@eq(%a, %b)
    : (tuple<i32,f64>, tuple<i32,f64>) -> i1
  return %res : i1
}

// CHECK: module {
// CHECK-LABEL: func.func @foo(%arg0: tuple<i32, f64>, %arg1: tuple<i32, f64>) -> i1 {
// CHECK-NEXT:   %0 = tuple.get %arg0, 0 : tuple<i32, f64> -> i32
// CHECK-NEXT:   %1 = tuple.get %arg1, 0 : tuple<i32, f64> -> i32
// CHECK-NEXT:   %2 = arith.cmpi eq, %0, %1 : i32
// CHECK-NEXT:   %3 = tuple.get %arg0, 1 : tuple<i32, f64> -> f64
// CHECK-NEXT:   %4 = tuple.get %arg1, 1 : tuple<i32, f64> -> f64
// CHECK-NEXT:   %5 = arith.cmpf oeq, %3, %4 : f64
// CHECK-NEXT:   %6 = arith.andi %2, %5 : i1
// CHECK-NEXT:   return %6 : i1
// CHECK-NEXT: }
// CHECK: }

// Ensure no trait IR remains
// CHECK-NOT: trait.
// CHECK-NOT: !trait.poly<
