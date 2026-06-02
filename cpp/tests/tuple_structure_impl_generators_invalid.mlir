// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(monomorphize-trait)" -verify-diagnostics

!T = !trait.poly<0>

trait.trait @tuple.HomogeneousTuple[!T] attributes {
  tuple.impl_generator = "homogeneous_tuple"
} {
  trait.assoc_type @Element
}

// expected-error @+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @homogeneous_mixed_rejected(
    %x: !trait.proj<@tuple.HomogeneousTuple[tuple<i64, i1>], "Element">)
    -> !trait.proj<@tuple.HomogeneousTuple[tuple<i64, i1>], "Element"> {
  return %x : !trait.proj<@tuple.HomogeneousTuple[tuple<i64, i1>], "Element">
}
