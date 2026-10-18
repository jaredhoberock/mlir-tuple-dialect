// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(canonicalize)" %s | FileCheck %s

// A tuple.make every operand of which is poison is poison of the tuple, a
// nested tuple's poison included, as a tuple.get of poison is poison of the
// element. The unit tuple has no component, so its make is the defined value
// its constant holds.
// CHECK-LABEL: func.func @make_of_poison
// CHECK-NOT: tuple.make
// CHECK-DAG: %[[FLAT:.*]] = ub.poison : tuple<i64, f32>
// CHECK-DAG: %[[NESTED:.*]] = ub.poison : tuple<i64, tuple<i64>>
// CHECK-DAG: %[[UNIT:.*]] = tuple.constant([]) : tuple<>
// CHECK: return %[[FLAT]], %[[NESTED]], %[[UNIT]]
func.func @make_of_poison() -> (tuple<i64, f32>, tuple<i64, tuple<i64>>, tuple<>) {
  %i = ub.poison : i64
  %f = ub.poison : f32
  %inner = ub.poison : tuple<i64>
  %flat = tuple.make(%i, %f : i64, f32) : tuple<i64, f32>
  %nested = tuple.make(%i, %inner : i64, tuple<i64>) : tuple<i64, tuple<i64>>
  %unit = tuple.make : tuple<>
  return %flat, %nested, %unit : tuple<i64, f32>, tuple<i64, tuple<i64>>, tuple<>
}
