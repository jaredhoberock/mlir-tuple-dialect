// RUN: mlir-opt --pass-pipeline="builtin.module(monomorphize-trait,convert-to-llvm)" %s | FileCheck %s

// -----
// CHECK-LABEL: llvm.func @cmp_eq_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_eq_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp eq, %a, %b : tuple<>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ne_empty
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ne_empty(%a : tuple<>, %b : tuple<>) -> i1 {
  %res = tuple.cmp ne, %a, %b : tuple<>
  return %res : i1
}

trait.trait @PartialEq {
  func.func private @eq(!trait.self, !trait.self) -> i1

  func.func @ne(%self: !trait.self, %other: !trait.self) -> i1 {
    %equal = trait.method.call @PartialEq::@eq<!trait.self>(%self, %other) : (!trait.self, !trait.self) -> i1 to (!trait.self, !trait.self) -> i1
    %true = arith.constant 1 : i1
    %result = arith.xori %equal, %true : i1
    return %result : i1
  }
}

trait.impl @PartialEq for i32 {
  func.func private @eq(%self: i32, %other: i32) -> i1 {
    %res = arith.cmpi eq, %self, %other : i32
    return %res : i1
  }
}

// -----
// CHECK-LABEL: llvm.func @cmp_eq_single_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_eq_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp eq, %a, %b : tuple<i32>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ne_single_i32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ne_single_i32(%a : tuple<i32>, %b : tuple<i32>) -> i1 {
  %res = tuple.cmp ne, %a, %b : tuple<i32>
  return %res : i1
}

trait.impl @PartialEq for i64 {
  func.func private @eq(%self: i64, %other: i64) -> i1 {
    %res = arith.cmpi eq, %self, %other : i64
    return %res : i1
  }
}

// -----
// CHECK-LABEL: llvm.func @cmp_eq_pair_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_eq_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp eq, %a, %b : tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ne_pair_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ne_pair_i64(%a : tuple<i64,i64>, %b : tuple<i64,i64>) -> i1 {
  %res = tuple.cmp ne, %a, %b : tuple<i64,i64>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_eq_nested_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_eq_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp eq, %a, %b : tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ne_nested_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ne_nested_i64(%a : tuple<i64,tuple<i64,i64>>, %b : tuple<i64,tuple<i64,i64>>) -> i1 {
  %res = tuple.cmp ne, %a, %b : tuple<i64,tuple<i64,i64>>
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_eq_mixed_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
!MixedTuple = tuple<tuple<>, i64, tuple<i64>>
func.func @cmp_eq_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp eq, %a, %b : !MixedTuple
  return %res : i1
}

// -----
// CHECK-LABEL: llvm.func @cmp_ne_mixed_i64
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: llvm.return
func.func @cmp_ne_mixed_i64(%a : !MixedTuple, %b : !MixedTuple) -> i1 {
  %res = tuple.cmp ne, %a, %b : !MixedTuple
  return %res : i1
}
