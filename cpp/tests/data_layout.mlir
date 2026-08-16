// RUN: mlir-opt --test-data-layout-query %s | FileCheck %s

// The builtin tuple answers data-layout queries with the standard non-packed
// struct layout, matching the LLVM literal struct the tuple lowers to. Each
// tuple query is paired with the struct it lowers to, and the two report the
// same alignment, bit size, preferred alignment, and size on every field --
// including the interior padding of a mixed-width tuple, whose i64 forces the
// i8 up to an 8-byte offset. The alignment answers are this pass's target-free
// defaults; what the pins guard is that a tuple and its struct never diverge.
// The layout model is attached when the tuple dialect loads, so a tuple.make
// anchors that load.

func.func @anchor_tuple_dialect(%a: i64) -> tuple<i64, i64> {
  %t = tuple.make(%a, %a : i64, i64) : tuple<i64, i64>
  return %t : tuple<i64, i64>
}

func.func @layout() {
  // CHECK: alignment = 4
  // CHECK-SAME: bitsize = 128
  // CHECK-SAME: preferred = 4
  // CHECK-SAME: size = 16
  // CHECK-SAME: tuple<i64, i64>
  "test.data_layout_query"() : () -> tuple<i64, i64>
  // CHECK: alignment = 4
  // CHECK-SAME: bitsize = 128
  // CHECK-SAME: preferred = 4
  // CHECK-SAME: size = 16
  // CHECK-SAME: llvm.struct<(i64, i64)>
  "test.data_layout_query"() : () -> !llvm.struct<(i64, i64)>

  // A nested tuple recurses through the same model.
  // CHECK: alignment = 4
  // CHECK-SAME: bitsize = 128
  // CHECK-SAME: preferred = 4
  // CHECK-SAME: size = 16
  // CHECK-SAME: tuple<i32, tuple<i64, i8>>
  "test.data_layout_query"() : () -> tuple<i32, tuple<i64, i8>>
  // CHECK: alignment = 4
  // CHECK-SAME: bitsize = 128
  // CHECK-SAME: preferred = 4
  // CHECK-SAME: size = 16
  // CHECK-SAME: llvm.struct<(i32, struct<(i64, i8)>)>
  "test.data_layout_query"() : () -> !llvm.struct<(i32, struct<(i64, i8)>)>

  // Interior padding matches: i8, then i64 at an 8-byte-aligned offset, then i16.
  // CHECK: alignment = 4
  // CHECK-SAME: bitsize = 128
  // CHECK-SAME: preferred = 4
  // CHECK-SAME: size = 16
  // CHECK-SAME: tuple<i8, i64, i16>
  "test.data_layout_query"() : () -> tuple<i8, i64, i16>
  // CHECK: alignment = 4
  // CHECK-SAME: bitsize = 128
  // CHECK-SAME: preferred = 4
  // CHECK-SAME: size = 16
  // CHECK-SAME: llvm.struct<(i8, i64, i16)>
  "test.data_layout_query"() : () -> !llvm.struct<(i8, i64, i16)>

  // The empty tuple lowers to an i8, so it reports that scalar's layout.
  // CHECK: alignment = 1
  // CHECK-SAME: bitsize = 8
  // CHECK-SAME: preferred = 1
  // CHECK-SAME: size = 1
  // CHECK-SAME: tuple<>
  "test.data_layout_query"() : () -> tuple<>
  return
}
