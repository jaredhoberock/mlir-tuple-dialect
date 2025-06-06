// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s

// -----

// CHECK: tuple.constant({{.*}}) : tuple<i64, i64, i64>
func.func @append_i64() -> tuple<i64,i64,i64>{
  %a = arith.constant 7 : i64
  %b = arith.constant 13 : i64
  %tup = tuple.constant(%a, %b : i64, i64) : tuple<i64,i64>
  %c = arith.constant 42 : i64
  %res = tuple.append %tup, %c : tuple<i64,i64>, i64 -> tuple<i64,i64,i64>
  return %res : tuple<i64,i64,i64>
}
