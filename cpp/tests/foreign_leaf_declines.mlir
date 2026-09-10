// The tuple converter lowers a tuple it owns to an LLVM struct, and declines a
// tuple carrying a type it does not own rather than aborting or reaching past it
// to force the inner type: a `!trait.poly` leaf survives the partial conversion
// intact, the function it sits in left standing for the step that owns the poly.
// A converter that overreached into the tuple to convert the poly, or aborted on
// it, would fail here.

// RUN: mlir-opt --pass-pipeline="builtin.module(convert-to-llvm)" %s | FileCheck %s

module {
  // CHECK: llvm.func @owns_its_tuple
  func.func @owns_its_tuple() {
    %c = tuple.constant([1 : i64, 2 : i64]) : tuple<i64, i64>
    return
  }
  // CHECK: func.func private @foreign_leaf
  // CHECK-SAME: tuple<i64, !trait.poly<0>>
  func.func private @foreign_leaf(%a: tuple<i64, !trait.poly<0>>) {
    return
  }
}
