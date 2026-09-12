// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s --pass-pipeline="builtin.module(tuple-elaborate)" | FileCheck %s

// `!tuple.poly<0>` is an occurrence of `!trait.poly<0>` constrained to a tuple.
// Instantiating the map body binds that parameter to the element type `i64`,
// which is not a tuple, so the occurrence spells nothing and stands as written.
// A substitution that walked into the occurrence instead would rebuild it around
// `i64` -- an occurrence wrapped around a type that is not a parameter at all.

!X = !trait.poly<0>
!T = !tuple.poly<0>

// CHECK-LABEL: func @f
// CHECK-NOT: tuple.map
// CHECK: %[[G:.+]] = tuple.get %arg0, 0 : tuple<i64> -> i64
// CHECK: %[[M:.+]] = tuple.make(%[[G]] : i64) : tuple<i64>
// CHECK: return %[[M]] : tuple<i64>
func.func @f(%xs: tuple<i64>, %t: !T) -> tuple<i64> {
  %res = tuple.map %xs : tuple<i64> -> tuple<i64> {
  ^bb0(%e: !X):
    %z = tuple.append %t, %e : !T, !X -> !T
    yield %e : !X
  }
  return %res : tuple<i64>
}
