// RUN: mlir-opt --test-data-layout-query %s | FileCheck %s

// A tuple's layout is the layout of the LLVM struct it lowers to, so a query
// builds LLVM types, and a dialect first loaded inside a pass aborts. The LLVM
// dialect loads with the tuple dialect, so a module that names no LLVM type is
// measured: the i64 follows the i1 at 8 bytes, and the tuple is 16. A
// tuple.make anchors the tuple dialect's load.

// CHECK: alignment = 8
// CHECK-SAME: size = 16
// CHECK-SAME: tuple<i1, i64>
module attributes {dlti.dl_spec = #dlti.dl_spec<i64 = dense<64> : vector<2xi64>>} {
  func.func @anchor_tuple_dialect(%a: i64) -> tuple<i64, i64> {
    %t = tuple.make(%a, %a : i64, i64) : tuple<i64, i64>
    return %t : tuple<i64, i64>
  }

  func.func @layout() {
    "test.data_layout_query"() : () -> tuple<i1, i64>
    return
  }
}
