// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(monomorphize-trait)" | FileCheck %s

!T = !trait.poly<0>

trait.trait @tuple.Tuple[!T] attributes {
  tuple.impl_generator = "tuple"
} {}

trait.trait @tuple.HomogeneousTuple[!T] attributes {
  tuple.impl_generator = "homogeneous_tuple"
} {
  trait.assoc_type @Element
}

// CHECK-LABEL: func.func @tuple_empty
// CHECK-NOT: trait.allege
func.func @tuple_empty() {
  %claim = trait.allege @tuple.Tuple[tuple<>]
  return
}

// CHECK-LABEL: func.func @tuple_mixed
// CHECK-NOT: trait.allege
func.func @tuple_mixed() {
  %claim = trait.allege @tuple.Tuple[tuple<i64, i1>]
  return
}

// CHECK-LABEL: func.func @homogeneous_empty
// CHECK-NOT: trait.allege
func.func @homogeneous_empty() {
  %claim = trait.allege @tuple.HomogeneousTuple[tuple<>]
  return
}

// CHECK-LABEL: func.func @homogeneous_singleton
// CHECK-NOT: trait.allege
func.func @homogeneous_singleton() {
  %claim = trait.allege @tuple.HomogeneousTuple[tuple<i64>]
  return
}

// CHECK-LABEL: func.func @homogeneous_pair
// CHECK-NOT: trait.allege
func.func @homogeneous_pair() {
  %claim = trait.allege @tuple.HomogeneousTuple[tuple<i64, i64>]
  return
}

// CHECK-LABEL: func.func @homogeneous_empty_element
// CHECK-SAME: (%{{.*}}: none) -> none
func.func @homogeneous_empty_element(
    %x: !trait.proj<@tuple.HomogeneousTuple[tuple<>], "Element">)
    -> !trait.proj<@tuple.HomogeneousTuple[tuple<>], "Element"> {
  return %x : !trait.proj<@tuple.HomogeneousTuple[tuple<>], "Element">
}

// CHECK-LABEL: func.func @homogeneous_singleton_element
// CHECK-SAME: (%{{.*}}: i64) -> i64
func.func @homogeneous_singleton_element(
    %x: !trait.proj<@tuple.HomogeneousTuple[tuple<i64>], "Element">)
    -> !trait.proj<@tuple.HomogeneousTuple[tuple<i64>], "Element"> {
  return %x : !trait.proj<@tuple.HomogeneousTuple[tuple<i64>], "Element">
}

// CHECK-LABEL: func.func @homogeneous_pair_element
// CHECK-SAME: (%{{.*}}: i64) -> i64
func.func @homogeneous_pair_element(
    %x: !trait.proj<@tuple.HomogeneousTuple[tuple<i64, i64>], "Element">)
    -> !trait.proj<@tuple.HomogeneousTuple[tuple<i64, i64>], "Element"> {
  return %x : !trait.proj<@tuple.HomogeneousTuple[tuple<i64, i64>], "Element">
}
