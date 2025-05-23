// RUN: opt %s | FileCheck %s

// ---- Test 1: single i64 ----

// CHECK-LABEL: func @constant_single
// CHECK: %[[C:.+]] = tuple.constant(%arg0 : i64) : tuple<i64>
func.func @constant_single(%arg0: i64) -> tuple<i64> {
  %c = tuple.constant(%arg0 : i64) : tuple<i64>
  return %c : tuple<i64>
}

// ---- Test 2: empty tuple ----

// CHECK-LABEL: func @constant_empty
// CHECK: %[[C:.+]] = tuple.constant : tuple<>
func.func @constant_empty() -> tuple<> {
  %c = tuple.constant : tuple<>
  return %c : tuple<>
}

// ---- Test 3: tuple of single ----

// CHECK-LABEL: func @constant_single_of_single
// CHECK: %[[C:.+]] = tuple.constant(%arg0 : tuple<i64>) : tuple<tuple<i64>>
func.func @constant_single_of_single(%arg0: tuple<i64>) -> tuple<tuple<i64>> {
  %c = tuple.constant(%arg0 : tuple<i64>) : tuple<tuple<i64>>
  return %c : tuple<tuple<i64>>
}

// ---- Test 4: pair of i64 ----

// CHECK-LABEL: func @constant_pair_of_i64
// CHECK: %[[C:.+]] = tuple.constant(%arg0, %arg1 : i64, i64) : tuple<i64, i64>
func.func @constant_pair_of_i64(%arg0: i64, %arg1: i64) -> tuple<i64,i64> {
  %c = tuple.constant(%arg0, %arg1 : i64, i64) : tuple<i64,i64>
  return %c : tuple<i64,i64>
}

// ---- Test 5: nested tuple (i64, tuple<i64,i64>) ----

// CHECK-LABEL: func @constant_nested
// CHECK: %[[C:.+]] = tuple.constant(%arg0, %arg1 : i64, tuple<i64, i64>) : tuple<i64, tuple<i64, i64>>
func.func @constant_nested(%arg0: i64, %arg1: tuple<i64,i64>) -> tuple<i64,tuple<i64,i64>> {
  %c = tuple.constant(%arg0, %arg1 : i64, tuple<i64,i64>) : tuple<i64,tuple<i64,i64>>
  return %c : tuple<i64,tuple<i64,i64>>
}

// ---- Test 6: nested in middle (i64, tuple<i64,i64>, i64) ----

// CHECK-LABEL: func @constant_nested_middle
// CHECK: %[[C:.+]] = tuple.constant(%arg0, %arg1, %arg2 : i64, tuple<i64, i64>, i64) : tuple<i64, tuple<i64, i64>, i64>
func.func @constant_nested_middle(%arg0: i64, %arg1: tuple<i64,i64>, %arg2: i64)
      -> tuple<i64,tuple<i64,i64>,i64> {
  %c = tuple.constant(%arg0, %arg1, %arg2 : i64, tuple<i64,i64>, i64)
        : tuple<i64,tuple<i64,i64>,i64>
  return %c : tuple<i64,tuple<i64,i64>,i64>
}

// ---- Test 7: deeply nested tuple ((i64,i64), tuple<i64>) ----

// CHECK-LABEL: func @constant_deeply_nested
// CHECK: %[[C:.+]] = tuple.constant(%arg0, %arg1 : tuple<i64, i64>, tuple<i64>) : tuple<tuple<i64, i64>, tuple<i64>>
func.func @constant_deeply_nested(%arg0: tuple<i64,i64>, %arg1: tuple<i64>)
      -> tuple<tuple<i64,i64>,tuple<i64>> {
  %c = tuple.constant(%arg0, %arg1 : tuple<i64,i64>, tuple<i64>)
        : tuple<tuple<i64,i64>,tuple<i64>>
  return %c : tuple<tuple<i64,i64>,tuple<i64>>
}

// ---- Test 8: mixed (tuple<>, i64, tuple<i64>) ----

// CHECK-LABEL: func @constant_mixed_tuple
// CHECK: %[[C:.+]] = tuple.constant(%arg0, %arg1, %arg2 : tuple<>, i64, tuple<i64>) : tuple<tuple<>, i64, tuple<i64>>
func.func @constant_mixed_tuple(
    %arg0: tuple<>,
    %arg1: i64,
    %arg2: tuple<i64>
) -> tuple<tuple<>,i64,tuple<i64>> {
  %c = tuple.constant(%arg0, %arg1, %arg2 : tuple<>, i64, tuple<i64>)
        : tuple<tuple<>,i64,tuple<i64>>
  return %c : tuple<tuple<>,i64,tuple<i64>>
}
