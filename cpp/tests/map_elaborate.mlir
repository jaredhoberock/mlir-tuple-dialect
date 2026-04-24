// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file --pass-pipeline="builtin.module(tuple-elaborate)" | FileCheck %s

// -----
// single i64

// CHECK-LABEL: func @map_i64_to_f32
// CHECK-NOT: tuple.map
// CHECK-NOT: tuple.yield
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<i64> -> i64
// CHECK: %[[F0:.+]] = arith.sitofp %[[G0]] : i64 to f32
// CHECK: %[[M:.+]] = tuple.make(%[[F0]] : f32) : tuple<f32>
// CHECK: return %[[M]] : tuple<f32>
func.func @map_i64_to_f32(%arg0: tuple<i64>) -> tuple<f32> {
  %res = tuple.map %arg0 : tuple<i64> -> tuple<f32> {
  ^bb0(%x: i64):
    %res = arith.sitofp %x : i64 to f32
    yield %res : f32
  }
  return %res : tuple<f32>
}

// -----
// pair of i64 — body instantiated once per element

// CHECK-LABEL: func @map_i64_i64_to_f32_f32
// CHECK-NOT: tuple.map
// CHECK-NOT: tuple.yield
// CHECK: %[[G0:.+]] = tuple.get %arg0, 0 : tuple<i64, i64> -> i64
// CHECK: %[[F0:.+]] = arith.sitofp %[[G0]] : i64 to f32
// CHECK: %[[G1:.+]] = tuple.get %arg0, 1 : tuple<i64, i64> -> i64
// CHECK: %[[F1:.+]] = arith.sitofp %[[G1]] : i64 to f32
// CHECK: %[[M:.+]] = tuple.make(%[[F0]], %[[F1]] : f32, f32) : tuple<f32, f32>
// CHECK: return %[[M]] : tuple<f32, f32>
func.func @map_i64_i64_to_f32_f32(%arg0: tuple<i64,i64>) -> tuple<f32,f32> {
  %res = tuple.map %arg0 : tuple<i64,i64> -> tuple<f32,f32> {
  ^bb0(%x: i64):
    %res = arith.sitofp %x : i64 to f32
    yield %res : f32
  }
  return %res : tuple<f32,f32>
}
