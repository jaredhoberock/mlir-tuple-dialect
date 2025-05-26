// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,convert-to-llvm)" %s | FileCheck %s

// -----
// CHECK-LABEL: llvm.func @cmp_lt_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_lt_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp lt, %a, %b : tuple<>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_le_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_le_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp le, %a, %b : tuple<>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_gt_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_gt_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp gt, %a, %b : tuple<>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ge_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ge_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp ge, %a, %b : tuple<>
  return %res : i1
}

trait.trait @PartialOrd {
  func.func private @lt(!trait.self, !trait.self) -> i1
  func.func private @le(!trait.self, !trait.self) -> i1
  func.func private @gt(!trait.self, !trait.self) -> i1
  func.func private @ge(!trait.self, !trait.self) -> i1
}

trait.impl @PartialOrd for i32 {
  func.func private @lt(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi slt, %self, %other : i32
    return %res : i1
  }
  func.func private @le(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi sle, %self, %other : i32
    return %res : i1
  }
  func.func private @gt(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi sgt, %self, %other : i32
    return %res : i1
  }
  func.func private @ge(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi sge, %self, %other : i32
    return %res : i1
  }
}

// -----
// CHECK-LABEL: llvm.func @cmp_lt_single_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_lt_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp lt, %a, %b : tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_le_single_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_le_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp le, %a, %b : tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_gt_single_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_gt_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp gt, %a, %b : tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ge_single_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ge_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp ge, %a, %b : tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_lt_pair_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_lt_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp lt, %a, %b : tuple<i64,i64>
  return %res : i1
}

trait.impl @PartialOrd for i64 {
  func.func private @lt(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi slt, %self, %other : i64
    return %res : i1
  }
  func.func private @le(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi sle, %self, %other : i64
    return %res : i1
  }
  func.func private @gt(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi sgt, %self, %other : i64
    return %res : i1
  }
  func.func private @ge(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi sge, %self, %other : i64
    return %res : i1
  }
}

// -----
// CHECK-LABEL: llvm.func @cmp_le_pair_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_le_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp le, %a, %b : tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_gt_pair_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_gt_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp gt, %a, %b : tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ge_pair_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ge_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp ge, %a, %b : tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_lt_nested_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_lt_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp lt, %a, %b : tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_le_nested_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_le_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp le, %a, %b : tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_gt_nested_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_gt_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp gt, %a, %b : tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ge_nested_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ge_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp ge, %a, %b : tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_lt_mixed_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
!MixedTuple = tuple<tuple<>, i64, tuple<i64>>
func.func @cmp_lt_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp lt, %a, %b : !MixedTuple
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_le_mixed_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_le_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp le, %a, %b : !MixedTuple
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_gt_mixed_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_gt_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp gt, %a, %b : !MixedTuple
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ge_mixed_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ge_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp ge, %a, %b : !MixedTuple
  return %res : i1
}
