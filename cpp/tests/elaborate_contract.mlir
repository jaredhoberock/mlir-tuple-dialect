// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Contract pin for `--tuple-elaborate`: after the pass, no higher-level
// tuple op or tuple.yield may remain, and no trait IR may be introduced by
// the elaboration itself.

// RUN: mlir-opt --pass-pipeline="builtin.module(tuple-elaborate)" %s | FileCheck %s

// CHECK-LABEL: func.func @contract_all_ops
// CHECK-NOT: tuple.map
// CHECK-NOT: tuple.foldl
// CHECK-NOT: tuple.flat_map
// CHECK-NOT: tuple.flatten
// CHECK-NOT: tuple.cat
// CHECK-NOT: tuple.append
// CHECK-NOT: tuple.drop_last
// CHECK-NOT: tuple.last
// CHECK-NOT: tuple.exclusive_scan
// CHECK-NOT: tuple.cmp
// CHECK-NOT: tuple.all
// CHECK-NOT: tuple.yield
// CHECK-NOT: trait.
// Positive anchor so an empty function body wouldn't pass the CHECK-NOTs vacuously.
// CHECK: return %{{.+}} : tuple<i32>
func.func @contract_all_ops(%t: tuple<i32,i32>, %u: tuple<i32,i32>,
                            %init: i32, %x: i32) -> tuple<i32> {
  // map
  %m = tuple.map %t : tuple<i32,i32> -> tuple<i32,i32> {
  ^bb0(%e: i32):
    %d = arith.addi %e, %e : i32
    yield %d : i32
  }

  // foldl
  %f = tuple.foldl %init, %m : i32, tuple<i32,i32> -> i32 {
  ^bb0(%acc: i32, %e: i32):
    %s = arith.addi %acc, %e : i32
    yield %s : i32
  }

  // flat_map
  %fm = tuple.flat_map %t : tuple<i32,i32> -> tuple<i32,i32,i32,i32> {
  ^bb0(%e: i32):
    %p = tuple.make(%e, %e : i32, i32) : tuple<i32,i32>
    yield %p : tuple<i32,i32>
  }

  // flatten (built from a nested tuple.make)
  %nn = tuple.make(%t, %u : tuple<i32,i32>, tuple<i32,i32>)
        : tuple<tuple<i32,i32>,tuple<i32,i32>>
  %fl = tuple.flatten %nn
        : tuple<tuple<i32,i32>,tuple<i32,i32>> -> tuple<i32,i32,i32,i32>

  // cat
  %c = tuple.cat %fm, %fl
       : tuple<i32,i32,i32,i32>, tuple<i32,i32,i32,i32>
       -> tuple<i32,i32,i32,i32,i32,i32,i32,i32>

  // append
  %a = tuple.append %t, %x : tuple<i32,i32>, i32 -> tuple<i32,i32,i32>

  // drop_last + last
  %dl = tuple.drop_last %a : tuple<i32,i32,i32> -> tuple<i32,i32>
  %ls = tuple.last %a : tuple<i32,i32,i32> -> i32

  // exclusive_scan — result arity is input arity + 1
  %es = tuple.exclusive_scan %t, %init : tuple<i32,i32>, i32 -> tuple<i32,i32,i32> {
  ^bb0(%acc: i32, %e: i32):
    %r = arith.addi %acc, %e : i32
    yield %r : i32
  }

  // cmp on empty tuples with explicit empty claims — folds to a constant
  // without leaving trait residue
  %ea = tuple.make : tuple<>
  %eb = tuple.make : tuple<>
  %eclaims = tuple.make : tuple<>
  %eq = tuple.cmp eq, %ea, %eb, %eclaims : tuple<>, tuple<>, tuple<>

  // all
  %bools = tuple.make(%eq : i1) : tuple<i1>
  %al = tuple.all %bools : tuple<i1> {
  ^bb0(%e: i1):
    tuple.yield %e : i1
  }

  // keep values live
  %keep = arith.addi %f, %ls : i32
  %out = tuple.make(%keep : i32) : tuple<i32>
  return %out : tuple<i32>
}
