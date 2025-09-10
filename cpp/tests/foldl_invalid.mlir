// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----
// non-tuple input

// expected-error @+2 {{'tuple.foldl' op input #0 must be a tuple type, got 'i64'}}
func.func @non_tuple_input(%arg0: i64, %arg1: i64) -> tuple<i64> {
  %res = tuple.foldl %arg0, %arg1 : i64, i64 -> i64 {
  ^bb0(%acc: i64, %x: i64):
    yield %acc: i64
  }
  return %res : i64
}

// -----
// arity mismatch

// expected-error @+2 {{'tuple.foldl' op arity mismatch: input #1 has arity 2 but a previous input has arity 1}}
func.func @arity_mismatch(%init: i64, %a: tuple<i64>, %b: tuple<i64, i64>) -> tuple<i64> {
  %res = tuple.foldl %init, %a, %b : i64, tuple<i64>, tuple<i64, i64> -> i64 {
  ^bb0(%acc: i64, %x: i64, %y: i64):
    yield %acc : i64
  }
  return %res : i64
}

// -----
// wrong number of block arguments

// expected-error @+2 {{'tuple.foldl' op body block must have 2 arguments (accumulator + one per input tuple), got 1}}
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

// -----
// non-tuple *second* input

// expected-error @+2 {{'tuple.foldl' op input #1 must be a tuple type, got 'i64'}}
func.func @non_tuple_second_input(%init: i64, %a: tuple<i64>, %b: i64) -> i64 {
  %res = tuple.foldl %init, %a, %b : i64, tuple<i64>, i64 -> i64 {
  ^bb0(%acc: i64, %x: i64, %y: i64):
    yield %acc : i64
  }
  return %res : i64
}

// -----
// unknown arity: non-acc arg must be purely polymorphic

!T = !tuple.poly<0>
!X = !trait.poly<0>
func.func @unknown_arity_non_pure_arg(%init: i32, %xs: !T) -> !T {
  // expected-error @+1 {{'tuple.foldl' op non-accumulator body argument #1 must be purely polymorphic (all leaves are e.g. '!trait.poly'); got 'i32'}}
  %res = tuple.foldl %init, %xs : i32, !T -> !T {
  ^bb0(%acc: !X, %e: i32):
    yield %acc : !X
  }
  return %res : !T
}

// -----
// unknown arity: init must fit accumulator formal

!T = !tuple.poly<0>
!A = !tuple.poly<1> // tuple-shaped formal for accumulator
!E = !trait.poly<0> // element formal for the the per-element arg
func.func @unknown_arity_bad_init(%init: i32, %xs: !T) -> !A {
  // expected-error @+1 {{'tuple.foldl' op type mismatch: expected a tuple type, but found 'i32'}}
  %res = tuple.foldl %init, %xs : i32, !T -> !A {
  ^bb0(%acc: !A, %e: !E):
    yield %acc : !A
  }
  return %res : !A
}

// -----
// unknown arity: closure check fails (accFormal != yieldFormal)

!T = !tuple.poly<0>
!A = !tuple.poly<1> // tuple-shaped accumulator formal
!E = !trait.poly<0>
func.func @unknown_arity_bad_closure(%init: !A, %xs: !T) -> !A {
  // expected-error @+1 {{'tuple.foldl' op type mismatch: expected a tuple type, but found 'i32'}}
  %res = tuple.foldl %init, %xs : !A, !T -> !A {
  ^bb0(%acc: !A, %e: !E):
    %c = arith.constant 0 : i32
    yield %c : i32        // yieldFormal incompatible with accFormal
  }
  return %res : !A
}

// -----
// unknown arity: op result formal must match accumulator formal

!T = !tuple.poly<0>
!E = !trait.poly<0>
func.func @unknown_arity_bad_result_formal(%init: i32, %xs: !T) -> f32 {
  // expected-error @+1 {{'tuple.foldl' op type mismatch: expected 'f32' but found 'i32'}}
  %res = tuple.foldl %init, %xs : i32, !T -> f32 {
  ^bb0(%acc: i32, %e: !E):
    yield %acc : i32
  }
  return %res : f32
}

// -----
// known arity: body element type must unify with actual element

func.func @known_arity_unify_fail(%init: i32, %xs: tuple<i32>) -> i32 {
  // expected-error @+1 {{'tuple.foldl' op type mismatch: expected 'f32' but found 'i32'}}
  %res = tuple.foldl %init, %xs : i32, tuple<i32> -> i32 {
  ^bb0(%acc: i32, %e: f32): // mismatch vs actual i32 element
    yield %acc : i32
  }
  return %res : i32
}


// -----
// known arity: final result must match threaded accumulator type

func.func @known_arity_bad_result(%init: i32, %xs: tuple<i32>) -> f32 {
  // expected-error @+1 {{'tuple.foldl' op type mismatch: expected 'f32' but found 'i32'}}
  %res = tuple.foldl %init, %xs : i32, tuple<i32> -> f32 {
  ^bb0(%acc: i32, %e: i32):
    yield %acc : i32
  }
  return %res : f32
}
