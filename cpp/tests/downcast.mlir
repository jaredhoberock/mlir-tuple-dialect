// RUN: mlir-opt %s | FileCheck %s

!T = !trait.poly<0>

trait.trait @Tuple[!T] {}

trait.trait @HomogeneousTuple[!T] where [@Tuple[!T]] {
  trait.assoc_type @Element
}

// CHECK-LABEL: func @downcast_basic
// CHECK: tuple.downcast
func.func @downcast_basic(
  %arg0: !trait.poly<0>,
  %arg1: !trait.claim<@Tuple[!trait.poly<0>]>
) -> !tuple.poly<0> {
  %res = tuple.downcast %arg0, %arg1 : !trait.poly<0>, !trait.claim<@Tuple[!trait.poly<0>]> -> !tuple.poly<0>
  return %res : !tuple.poly<0>
}

// CHECK-LABEL: func @downcast_nested
// CHECK: tuple.downcast
func.func @downcast_nested(
  %arg0: tuple<!trait.poly<0>, tuple<!trait.poly<1>, !trait.poly<2>>>,
  %arg1: !trait.claim<@Tuple[!trait.poly<1>]>
) -> tuple<!trait.poly<0>, tuple<!tuple.poly<1>, !trait.poly<2>>> {
  %res = tuple.downcast %arg0, %arg1
    : tuple<!trait.poly<0>, tuple<!trait.poly<1>, !trait.poly<2>>>, !trait.claim<@Tuple[!trait.poly<1>]>
    -> tuple<!trait.poly<0>, tuple<!tuple.poly<1>, !trait.poly<2>>>
  return %res : tuple<!trait.poly<0>, tuple<!tuple.poly<1>, !trait.poly<2>>>
}

// CHECK-LABEL: func @downcast_monomorphic_identity
// CHECK: tuple.downcast
func.func @downcast_monomorphic_identity(
  %arg0: tuple<i64>,
  %arg1: !trait.claim<@Tuple[tuple<i64>]>
) -> tuple<i64> {
  %res = tuple.downcast %arg0, %arg1 : tuple<i64>, !trait.claim<@Tuple[tuple<i64>]> -> tuple<i64>
  return %res : tuple<i64>
}

// CHECK-LABEL: func @downcast_from_homogeneous_projection
// CHECK: trait.project
// CHECK: tuple.downcast
func.func @downcast_from_homogeneous_projection(
  %arg0: !trait.poly<0>,
  %arg1: !trait.claim<@HomogeneousTuple[!trait.poly<0>]>
) -> !tuple.poly<0> {
  %tuple_claim = trait.project %arg1 : @HomogeneousTuple[!trait.poly<0>] to @Tuple[!trait.poly<0>]
  %res = tuple.downcast %arg0, %tuple_claim : !trait.poly<0>, !trait.claim<@Tuple[!trait.poly<0>]> -> !tuple.poly<0>
  return %res : !tuple.poly<0>
}
