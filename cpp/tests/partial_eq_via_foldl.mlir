// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,inline)" %s | FileCheck %s

!S = !trait.poly<0>
!O = !trait.poly<1>
trait.trait @PartialEq[!S,!O] {
  func.func private @eq(!S,!O) -> i1
}

trait.impl for @PartialEq[i32,i32] {
  func.func @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    return %res : i1
  }
}

trait.impl for @PartialEq[f64,f64] {
  func.func @eq(%self: f64, %other: f64) -> i1 {
    %res = arith.cmpf oeq, %self, %other : f64
    return %res : i1
  }
}

// this trait maps the PartialEq trait over a tuple's element types
// and provides a tuple of their claims
!MapPartialEqS = !trait.poly<2>
!MapPartialEqO = !trait.poly<3>
!MapPartialEqC = !trait.poly<4>
trait.trait @tuple.MapPartialEq[!MapPartialEqS,!MapPartialEqO,!MapPartialEqC] attributes {
  tuple.impl_generator = "map",
  tuple.mapped_trait = @PartialEq
} {
  func.func private @claims() -> !MapPartialEqC
}

// this is the polymorphic tuple impl of PartialEq
!TS = !tuple.poly<5>
!TO = !tuple.poly<6>
!TC = !tuple.poly<7>
!ES = !trait.poly<8>
!EO = !trait.poly<9>
trait.impl @tuple.PartialEq for @PartialEq[!TS,!TO] where [
  @tuple.MapPartialEq[!TS,!TO,!TC]
] {
  func.func @eq(%self: !TS, %other: !TO) -> i1 {
    // first get a tuple of elementwise PartialEq claims
    %a = trait.assume @tuple.MapPartialEq[!TS,!TO,!TC]
    %claims = trait.method.call %a @tuple.MapPartialEq[!TS,!TO,!TC]::@claims()
      : () -> !TC

    // fold @PartialEq::@eq over the tuples
    %init = arith.constant 1 : i1

    %res = tuple.foldl %init, %self, %other, %claims : i1, !TS, !TO, !TC -> i1 {
    ^bb0(%acc: i1, %s: !ES, %o: !EO, %c: !trait.claim<@PartialEq[!ES,!EO]>):
      %eq = trait.method.call %c @PartialEq[!ES,!EO]::@eq(%s, %o)
        : (!ES,!EO) -> i1
      %res = arith.andi %acc, %eq : i1
      yield %res : i1
    }
    return %res : i1
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
