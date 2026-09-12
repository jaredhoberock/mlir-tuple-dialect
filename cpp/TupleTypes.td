// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TUPLE_TYPES_TD
#define TUPLE_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Tuple.td"
include "TraitTypeInterfaces.td"

def PolyType : TypeDef<TupleDialect, "Poly", [
  GenericTypeInterface,
  PolymorphicTypeInterface
]> {
  let mnemonic = "poly";
  let summary = "Placeholder type representing a polymorphic tuple type";
  let description = [{
    The `!tuple.poly` type represents a universally quantified type parameter
    that is also a tuple. It is used to model the parameters of generic functions
    on tuples.

    `!tuple.poly` is an occurrence of the `!trait.poly` it wraps that constrains
    that parameter to a tuple: a declaration binds the inner parameter, and a
    substitution carrying it to a type that is not tuple-like spells nothing
    here, which leaves the constraint standing and the comparison that reads it
    refuses.

    The label it prints is the inner parameter's, and it is written the same way:
    `!tuple.poly<0>` is `!tuple.poly<!trait.poly<0>>`, the declaration's first
    parameter constrained to a tuple. Labels are non-negative.
  }];

  let extraClassDeclaration = [{
    // for GenericTypeInterface
    Type specializeWith(const ::mlir::trait::SpecializationMap &subst) const;
    Type getParameterAtom() const { return getInner(); }

    // for PolymorphicTypeInterface
    bool isPolymorphic() const { return true; }
  }];

  let parameters = (ins "::mlir::trait::PolyType":$inner);

  let hasCustomAssemblyFormat = 1;
}

#endif // TUPLE_TYPES_TD
