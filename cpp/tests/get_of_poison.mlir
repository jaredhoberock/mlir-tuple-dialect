// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --pass-pipeline="builtin.module(canonicalize)" %s | FileCheck %s

// A tuple.get of a poison tuple folds to poison of the element's type, a leaf
// such as a memref or a tuple alike, so a read of an aggregate no path defines
// leaves no tuple behind.
// CHECK-LABEL: func.func @get_of_poison
// CHECK-NOT: tuple.get
// CHECK-DAG: %[[LEAF:.*]] = ub.poison : memref<?xf32>
// CHECK-DAG: %[[INNER:.*]] = ub.poison : tuple<i64, f32>
// CHECK: return %[[LEAF]], %[[INNER]] : memref<?xf32>, tuple<i64, f32>
func.func @get_of_poison() -> (memref<?xf32>, tuple<i64, f32>) {
  %p = ub.poison : tuple<memref<?xf32>, tuple<i64, f32>>
  %leaf = tuple.get %p, 0 : tuple<memref<?xf32>, tuple<i64, f32>> -> memref<?xf32>
  %inner = tuple.get %p, 1 : tuple<memref<?xf32>, tuple<i64, f32>> -> tuple<i64, f32>
  return %leaf, %inner : memref<?xf32>, tuple<i64, f32>
}
