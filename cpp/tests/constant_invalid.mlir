// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// The verifier carries the leaf cross-check: the value's arity must be the
// tuple's, and each scalar leaf attribute's type must be the element's.

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @arity_mismatch() -> tuple<i64, i64> {
  // expected-error@+1 {{constant element count 1 does not match tuple arity 2}}
  %c = tuple.constant([1 : i64]) : tuple<i64, i64>
  return %c : tuple<i64, i64>
}

// -----

func.func @scalar_type_mismatch() -> tuple<i64> {
  // expected-error@+1 {{constant attribute 1 : i32 is not of scalar type 'i64'}}
  %c = tuple.constant([1 : i32]) : tuple<i64>
  return %c : tuple<i64>
}

// -----

func.func @nested_element_not_array() -> tuple<tuple<i64>> {
  // expected-error@+1 {{tuple constant requires an array attribute}}
  %c = tuple.constant([1 : i64]) : tuple<tuple<i64>>
  return %c : tuple<tuple<i64>>
}
