// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// non-tuple input

// expected-error @+2 {{'tuple.map' op input #0 must be a 'tuple', got 'i64'}}
func.func @non_tuple_input(%arg0: i64) -> tuple<i64> {
  %res = tuple.map %arg0 : i64 -> tuple<i64> {
  ^bb0(%x: i64):
    yield %x : i64
  }
  return %res : tuple<i64>
}

// -----
// arity mismatch

// expected-error @+2 {{'tuple.map' op all input tuples must have the same arity, expected 1 but input #1 has arity 2}}
func.func @arity_mismatch(%a: tuple<i64>, %b: tuple<i64, i64>) -> tuple<i64> {
  %res = tuple.map %a, %b : tuple<i64>, tuple<i64, i64> -> tuple<i64> {
  ^bb0(%x: i64, %y: i64):
    yield %x : i64
  }
  return %res : tuple<i64>
}

// -----
// result is not a tuple

// expected-error @+2 {{'tuple.map' op result must be a tuple type, got 'i64'}}
func.func @non_tuple_result(%arg0: tuple<i64>) -> i64 {
  %res = tuple.map %arg0 : tuple<i64> -> i64 {
  ^bb0(%x: i64):
    yield %x : i64
  }
  return %res : i64
}

// -----
// result arity mismatch

// expected-error @+2 {{'tuple.map' op result tuple must have the same arity as input, expected 1 but result tuple has arity 2}}
func.func @result_arity_mismatch(%arg0: tuple<i32>) -> tuple<i32,i32> {
  %res = tuple.map %arg0 : tuple<i32> -> tuple<i32,i32> {
  ^bb0(%x: i32):
    yield %x : i32
  }
  return %res : tuple<i32,i32>
}

// -----
// wrong number of block arguments

// expected-error @+2 {{'tuple.map' op body block must have 2 arguments to match the number of tuple inputs, got 1}}
func.func @bad_block_arg_count(%a: tuple<i64>, %b: tuple<i64>) -> tuple<i64> {
  %res = tuple.map %a, %b : tuple<i64>, tuple<i64> -> tuple<i64> {
  ^bb0(%x: i64):
    yield %x : i64
  }
  return %res : tuple<i64>
}

// -----
// missing yield

// expected-error @+2 {{'tuple.map' op body block must terminate with `tuple.yield`, got func.return}}
func.func @missing_yield(%arg0: tuple<i64>) -> tuple<i64> {
  %res = tuple.map %arg0 : tuple<i64> -> tuple<i64> {
  ^bb0(%x: i64):
    func.return
  }
  return %res : tuple<i64>
}
