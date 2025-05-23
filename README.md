# mlir-tuple-dialect

An **MLIR** dialect that adds first-class support for statically-typed tuples and structural operations on them. It integrates with `mlir-trait-dialect` to enable **trait-bounded generic operations** on tuple types.

---
## Why?
Many generic algorithms in Rust or C++ operate over tuple-like product types and assume trait bounds on the elements (e.g., "every element must be `PartialEq`"). To encode this structure in MLIR, we need:

- Concrete tuple types: `tuple<i32, i32>`
- Symbolic types with elementwise trait bounds: `!tuple.any<[@PartialEq]>`
- Structural ops like `tuple.get`, `tuple.constant`, and `tuple.cmp`

This dialect provides those pieces, and supports **lowering to LLVM** via flattening to `vector<NxiN>` for homogeneous tuples.

---
## Tiny example

The snippet below defines a generic `PartialEq` implementation for all tuples whose elements implement `PartialEq`, and uses `tuple.cmp` to compare them:

<details>
<summary>Click to expand MLIR</summary>

```mlir
trait.trait @PartialEq {
  func.func private @eq(!trait.self, !trait.self) -> i1
}

// generic impl for any tuple whose elements have PartialEq
!T = !tuple.any<[@PartialEq]>
trait.impl @PartialEq for !T {
  func.func private @eq(%a: !T, %b: !T) -> i1 {
    %res = tuple.cmp eq, %a, %b : !T
    return %res : i1
  }
}

// call the generic impl with a concrete type
func.func @check(%x: tuple<i32, i32>, %y: tuple<i32, i32>) -> i1 {
  %r = trait.method.call @PartialEq::@eq<tuple<i32, i32>>(%x, %y)
       : (!trait.self, !trait.self) -> i1 to (tuple<i32, i32>, tuple<i32, i32>) -> i1
  return %r : i1
}
```
</details>

<details>
<summary>Lowered to LLVM</summary>
```
%0 = llvm.extractelement %x[0]
%1 = llvm.extractelement %y[0]
%2 = llvm.call @__trait_PartialEq_impl_i32_eq(%0, %1)
%3 = llvm.extractelement %x[1]
%4 = llvm.extractelement %y[1]
%5 = llvm.call @__trait_PartialEq_impl_i32_eq(%3, %4)
%6 = llvm.and %2, %5
llvm.return %6
```
</details>
