// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt --inline %s | FileCheck %s

// A multi-block (diamond) callee cannot be inlined into a tuple.map body: the
// region is SizedRegion<1>, so splicing three blocks in would leave a region
// that fails to verify. The inliner interface refuses on the source's block
// count, and the call stays.
// CHECK-LABEL: func.func @caller
// CHECK: tuple.map
// CHECK: func.call @diamond
func.func private @diamond(%x: i64, %p: i1) -> f32 {
  cf.cond_br %p, ^bb1, ^bb2
^bb1:
  %a = arith.sitofp %x : i64 to f32
  cf.br ^bb3(%a : f32)
^bb2:
  %b = arith.constant 0.0 : f32
  cf.br ^bb3(%b : f32)
^bb3(%r: f32):
  return %r : f32
}
func.func @caller(%arg0: tuple<i64>, %p: i1) -> tuple<f32> {
  %res = tuple.map %arg0 : tuple<i64> -> tuple<f32> {
  ^bb0(%x: i64):
    %c = func.call @diamond(%x, %p) : (i64, i1) -> f32
    yield %c : f32
  }
  return %res : tuple<f32>
}
