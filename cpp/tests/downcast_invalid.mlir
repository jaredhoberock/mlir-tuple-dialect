// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// claim for one variable cannot downcast another

!T = !trait.poly<0>
trait.trait @Tuple[!T] {}

func.func @wrong_value_poly(
  %arg0: !trait.poly<1>,
  %arg1: !trait.claim<@Tuple[!trait.poly<0>]>
) -> !tuple.poly<1> {
  // expected-error @+1 {{value type does not contain the claimed tuple-polymorphic type}}
  %res = tuple.downcast %arg0, %arg1 : !trait.poly<1>, !trait.claim<@Tuple[!trait.poly<0>]> -> !tuple.poly<1>
  return %res : !tuple.poly<1>
}

// -----
// the result must preserve the rewritten variable identity

!T = !trait.poly<0>
trait.trait @Tuple[!T] {}

func.func @wrong_result_poly(
  %arg0: !trait.poly<0>,
  %arg1: !trait.claim<@Tuple[!trait.poly<0>]>
) -> !tuple.poly<1> {
  // expected-error @+1 {{result type must be the value type with}}
  %res = tuple.downcast %arg0, %arg1 : !trait.poly<0>, !trait.claim<@Tuple[!trait.poly<0>]> -> !tuple.poly<1>
  return %res : !tuple.poly<1>
}

// -----
// polymorphic downcast must actually introduce a tuple-polymorphic view

!T = !trait.poly<0>
trait.trait @Tuple[!T] {}

func.func @no_rewrite(
  %arg0: !trait.poly<0>,
  %arg1: !trait.claim<@Tuple[!trait.poly<0>]>
) -> !trait.poly<0> {
  // expected-error @+1 {{result type must be the value type with}}
  %res = tuple.downcast %arg0, %arg1 : !trait.poly<0>, !trait.claim<@Tuple[!trait.poly<0>]> -> !trait.poly<0>
  return %res : !trait.poly<0>
}

// -----
// concrete tuple claims are only valid as identity conversions

!T = !trait.poly<0>
trait.trait @Tuple[!T] {}

func.func @concrete_claim_value_mismatch(
  %arg0: tuple<i64>,
  %arg1: !trait.claim<@Tuple[tuple<i32>]>
) -> tuple<i32> {
  // expected-error @+1 {{value type must match concrete claim type argument}}
  %res = tuple.downcast %arg0, %arg1 : tuple<i64>, !trait.claim<@Tuple[tuple<i32>]> -> tuple<i32>
  return %res : tuple<i32>
}

// -----
// tuple-structure claims must name exactly the tuple type being proven

!T = !trait.poly<0>
trait.trait @Tuple[!T] {}

func.func @tuple_claim_wrong_arity(
  %arg0: !trait.poly<0>,
  %arg1: !trait.claim<@Tuple[!trait.poly<0>, !trait.poly<1>]>
) -> !tuple.poly<0> {
  // expected-error @+1 {{tuple-structure claim must have exactly one type argument}}
  %res = tuple.downcast %arg0, %arg1 : !trait.poly<0>, !trait.claim<@Tuple[!trait.poly<0>, !trait.poly<1>]> -> !tuple.poly<0>
  return %res : !tuple.poly<0>
}

// -----
// polymorphic tuple claims must identify one trait-polymorphic variable

!T = !trait.poly<0>
!P = tuple<!trait.poly<0>>
trait.trait @Tuple[!T] {}

func.func @tuple_claim_non_bare_poly(
  %arg0: !P,
  %arg1: !trait.claim<@Tuple[!P]>
) -> tuple<!tuple.poly<0>> {
  // expected-error @+1 {{polymorphic tuple-structure claim must name one !trait.poly}}
  %res = tuple.downcast %arg0, %arg1 : !P, !trait.claim<@Tuple[!P]> -> tuple<!tuple.poly<0>>
  return %res : tuple<!tuple.poly<0>>
}

// -----
// internal tuple helper facts are not accepted by tuple.downcast

func.func @internal_tuple_claim(
  %arg0: !trait.poly<0>,
  %arg1: !trait.claim<@tuple.Tuple[!trait.poly<0>]>
) -> !tuple.poly<0> {
  // expected-error @+1 {{claim must be for a tuple-structure trait}}
  %res = tuple.downcast %arg0, %arg1 : !trait.poly<0>, !trait.claim<@tuple.Tuple[!trait.poly<0>]> -> !tuple.poly<0>
  return %res : !tuple.poly<0>
}

// -----
// homogeneous-tuple facts must be projected to Tuple before downcast

func.func @homogeneous_tuple_claim(
  %arg0: !trait.poly<0>,
  %arg1: !trait.claim<@HomogeneousTuple[!trait.poly<0>]>
) -> !tuple.poly<0> {
  // expected-error @+1 {{claim must be for a tuple-structure trait}}
  %res = tuple.downcast %arg0, %arg1 : !trait.poly<0>, !trait.claim<@HomogeneousTuple[!trait.poly<0>]> -> !tuple.poly<0>
  return %res : !tuple.poly<0>
}

// -----
// tuple.downcast requires the source Tuple trait symbol to be present

func.func @missing_tuple_trait(
  %arg0: !trait.poly<0>,
  %arg1: !trait.claim<@Tuple[!trait.poly<0>]>
) -> !tuple.poly<0> {
  // expected-error @+1 {{couldn't find trait.trait '@Tuple'}}
  %res = tuple.downcast %arg0, %arg1 : !trait.poly<0>, !trait.claim<@Tuple[!trait.poly<0>]> -> !tuple.poly<0>
  return %res : !tuple.poly<0>
}
