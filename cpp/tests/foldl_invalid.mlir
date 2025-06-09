// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// non-tuple input

// expected-error @+2 {{'tuple.foldl' op operand #1 must be any concrete tuple type, but got 'i64'}}
func.func @non_tuple_input(%arg0: i64, %arg1: i64) -> tuple<i64> {
  %res = tuple.foldl %arg0, %arg1 : i64, i64 -> i64 {
  ^bb0(%acc: i64, %x: i64):
    yield %acc: i64
  }
  return %res : i64
}

// -----
// wrong number of block arguments

// expected-error @+2 {{'tuple.foldl' op body block must have 2 arguments, got 1}}
func.func @bad_block_arg_count(%init: i64, %tuple: tuple<i64>) -> i64 {
  %res = tuple.foldl %init, %tuple : i64, tuple<i64> -> i64 {
  ^bb0(%acc: i64):
    yield %acc : i64
  }
  return %res : i64
}

// -----
// missing yield

// expected-error @+2 {{'tuple.foldl' op body block must terminate with `tuple.yield`, got func.return}}
func.func @missing_yield(%init: i64, %tuple: tuple<i64>) -> tuple<i64> {
  %res = tuple.foldl %init, %tuple : i64, tuple<i64> -> i64 {
  ^bb0(%acc: i64, %e: i64):
    func.return
  }
  return %res : i64
}
