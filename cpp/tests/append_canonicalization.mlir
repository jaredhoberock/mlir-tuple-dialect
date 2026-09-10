// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// -----

// The inputs are block arguments so the fused result stays a tuple.make: a
// tuple.make of all-constant operands would fold further, to a tuple.constant.
// CHECK: tuple.make({{.*}}) : tuple<i64, i64, i64>
func.func @append_i64(%a: i64, %b: i64, %c: i64) -> tuple<i64,i64,i64>{
  %tup = tuple.make(%a, %b : i64, i64) : tuple<i64,i64>
  %res = tuple.append %tup, %c : tuple<i64,i64>, i64 -> tuple<i64,i64,i64>
  return %res : tuple<i64,i64,i64>
}
