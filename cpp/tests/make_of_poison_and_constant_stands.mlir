// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(canonicalize)" %s | FileCheck %s

// A tuple.make of a poison and a constant is no tuple.constant: a constant's
// value holds no poison, so the construction stands, its poison component
// lowering as a poison operand of the struct it builds.
// CHECK-LABEL: func.func @make_of_poison_and_constant
// CHECK-NOT: tuple.constant
// CHECK-DAG: %[[P:.*]] = ub.poison : i64
// CHECK-DAG: %[[C:.*]] = arith.constant 5 : i64
// CHECK: %[[T:.*]] = tuple.make(%[[P]], %[[C]] : i64, i64) : tuple<i64, i64>
// CHECK: return %[[T]]
func.func @make_of_poison_and_constant() -> tuple<i64, i64> {
  %p = ub.poison : i64
  %c = arith.constant 5 : i64
  %t = tuple.make(%p, %c : i64, i64) : tuple<i64, i64>
  return %t : tuple<i64, i64>
}

// The same when the poison component is read from a poison tuple, whose
// tuple.get folds to poison of the element: the tuple taken apart and made
// again with a constant stays a construction.
// CHECK-LABEL: func.func @remade_component_of_poison
// CHECK-NOT: tuple.constant
// CHECK-NOT: tuple.get
// CHECK: tuple.make
func.func @remade_component_of_poison() -> tuple<i64, i64> {
  %pt = ub.poison : tuple<i64, i64>
  %p = tuple.get %pt, 0 : tuple<i64, i64> -> i64
  %c = arith.constant 5 : i64
  %t = tuple.make(%p, %c : i64, i64) : tuple<i64, i64>
  return %t : tuple<i64, i64>
}
