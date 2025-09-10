// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// non-tuple input

// expected-error @+2 {{'tuple.map' op input #0 must be a tuple type, got 'i64'}}
func.func @non_tuple_input(%arg0: i64) -> tuple<i64> {
  %res = tuple.map %arg0 : i64 -> tuple<i64> {
  ^bb0(%x: i64):
    yield %x : i64
  }
  return %res : tuple<i64>
}

// -----
// arity mismatch

// expected-error @+2 {{'tuple.map' op arity mismatch: input #1 has arity 2 but a previous input has arity 1}}
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

// expected-error @+2 {{'tuple.map' op arity mismatch: result tuple has arity 2, but input tuples have arity 1}}
func.func @result_arity_mismatch(%arg0: tuple<i32>) -> tuple<i32,i32> {
  %res = tuple.map %arg0 : tuple<i32> -> tuple<i32,i32> {
  ^bb0(%x: i32):
    yield %x : i32
  }
  return %res : tuple<i32,i32>
}

// -----
// wrong number of block arguments

// expected-error @+2 {{'tuple.map' op body block must have 2 arguments (one per input tuple), got 1}}
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

// -----
// unknown arity: result must be !tuple.poly

!T = !tuple.poly<0>
!X = !trait.poly<0>
func.func @unknown_arity_bad_result(%xs: !T) -> tuple<i32> {
  // expected-error @+1 {{'tuple.map' op result must be !tuple.poly when arity is unknown}}
  %res = tuple.map %xs : !T -> tuple<i32> {
  ^bb0(%x: !X):
    yield %x : !X
  }
  return %res : tuple<i32>
}

// -----
// unknown arity: body arg must be purely polymorphic

!T = !tuple.poly<0>
!R = !tuple.poly<1>
func.func @unknown_arity_nonpoly_arg(%xs: !T) -> !R {
  // expected-error @+1 {{'tuple.map' op body argument #0 must be purely polymorphic}}
  %res = tuple.map %xs : !T -> !R {
  ^bb0(%x: i32):
    yield %x : i32
  }
  return %res : !R
}

// -----
// unknown arity: yield/result must be purely polymorphic

!T = !tuple.poly<0>
!R = !tuple.poly<1>
!X = !trait.poly<0>
func.func @unknown_arity_nonpoly_yield(%xs: !T) -> !R {
  // expected-error @+1 {{'tuple.map' op body yield/result must be purely polymorphic}}
  %res = tuple.map %xs : !T -> !R {
  ^bb0(%x: !X):
    %c0 = arith.constant 0 : i32
    yield %c0 : i32
  }
  return %res : !R
}

// -----
// one input (tuple<>): body must have 1 arg

func.func @empty_tuple_wrong_body_arity(%xs: tuple<>) -> tuple<> {
  // expected-error @+1 {{'tuple.map' op body block must have 1 arguments (one per input tuple), got 0}}
  %res = tuple.map %xs : tuple<> -> tuple<> {
  ^bb0():                // <-- should be 1 arg, not 0
    %c = arith.constant 0 : i1
    yield %c : i1
  }
  return %res : tuple<>
}
