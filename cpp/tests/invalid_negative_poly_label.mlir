// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// A label names a position in the declaration that binds it, so it is
// non-negative. A `!tuple.poly` prints and parses its inner parameter's label,
// so it refuses a negative one where it is written.

// expected-error @below {{a !trait.poly label is non-negative; found -1}}
!T = !tuple.poly<-1>
func.func @negative_label(%a: !T) -> !T {
  return %a : !T
}

// -----

// The nested spelling names the same parameter and refuses the same label.

// expected-error @below {{a !trait.poly label is non-negative; found -2}}
!U = !tuple.poly<!trait.poly<-2>>
func.func @negative_label_nested(%a: !U) -> !U {
  return %a : !U
}
