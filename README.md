# mlir-tuple-dialect

An **MLIR dialect** that adds first-class support for tuples and structural operations on them.  
It integrates with [`mlir-trait-dialect`](../mlir-trait-dialect) to enable **generic, trait-bounded operations** on tuple types, both concrete and polymorphic.

---

## Why?

Many generic algorithms in Rust or C++ operate over tuple-like product types with trait bounds on elements (e.g. “all elements implement `PartialEq`” or “all elements implement `PartialOrd`”).  
To encode this structure in MLIR, we need:

- **Concrete tuple types**:  
  ```mlir
  tuple<i32, tuple<i1, f64>>
  ```

- **Polymorphic tuple types** parameterized by type variables:
  ```mlir
  !tuple.poly<0>
  ```

- **Structural operations** such as:
  - `tuple.make` - construct a tuple
  - `tuple.get` - project a tuple element
  - `tuple.map` / `tuple.foldl` - iterate structurally over tuple elements
  - `tuple.cmp` - lexicographic comparison given elementwise trait claims

This dialect provides those primitives, and works hand-in-hand with the trait dialect to enable both monomorphic and polymorphic code to be expressed naturally and lowered systematically.

---

## Example: tuple equality

The snippet below shows a call to `tuple.cmp eq` on a pair of `i64`s.
During impl resolution, the tuple dialect generates the necessary impls of `PartialEq` for tuples, relying on an elementwise impl of `PartialEq[i64,i64]`.

```mlir
trait.trait @PartialEq[!S,!O] {
  func.func private @eq(!S, !O) -> i1
}

trait.impl for @PartialEq[i64,i64] {
  func.func private @eq(%a: i64, %b: i64) -> i1 {
    %res = arith.cmpi eq, %a, %b : i64
    return %res : i1
  }
}

func.func @eq_i64_pair(%x: tuple<i64,i64>, %y: tuple<i64,i64>) -> i1 {
  %res = tuple.cmp eq, %x, %y : tuple<i64,i64>, tuple<i64,i64>
  return %res : i1
}
```

After impl resolution and monomorphization, the operation is lowered into elementwise comparisons:

```mlir
func.func @eq_i64_pair(%x: tuple<i64,i64>, %y: tuple<i64,i64>) -> i1 {
  %0 = tuple.get %x, 0 : tuple<i64,i64> -> i64
  %1 = tuple.get %y, 0 : tuple<i64,i64> -> i64
  %2 = arith.cmpi eq, %0, %1 : i64

  %3 = tuple.get %x, 1 : tuple<i64,i64> -> i64
  %4 = tuple.get %y, 1 : tuple<i64,i64> -> i64
  %5 = arith.cmpi eq, %3, %4 : i64

  %6 = arith.andi %2, %5 : i1
  return %6 : i1
}
```
