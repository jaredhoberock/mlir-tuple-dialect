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

    The `unique_id` serves as a unique identifier for this type within the module.

    Alternatively, `unique` can be specified instead of `unique_id` to request a 
    `!tuple.poly` with an automatically-generated `unique_id`. Unique IDs are
    allocated in monotonically decreasing order beginning at `-1`.

    `!tuple.poly` is identical to `!trait.poly` *except* that it instantiates
    into `!tuple.infer`. A `!tuple.infer` type will unify only with a tuple-like type.
  }];

  let extraClassDeclaration = [{
    // create a PolyType with an unused unique_id
    static PolyType getUnique(MLIRContext* ctx);

    // for GenericTypeInterface
    Type instantiate(DenseMap<Type,Type> &inst, uint64_t &idCounter);
    Type specializeWith(const DenseMap<Type,Type> &subst) const;

    // for PolymorphicTypeInterface
    bool isPolymorphic() const { return true; }
  }];

  let parameters = (ins "::mlir::trait::PolyType":$inner);

  let hasCustomAssemblyFormat = 1;
}

// Represents an inference variable used during type inference
def InferenceType : TypeDef<TupleDialect, "Inference", [
  PolymorphicTypeInterface,
  UnificationTypeInterface
]> {
  let mnemonic = "infer";
  let summary = "INTERNAL: Type inference variable (should not appear in final IR)";

  let description = [{
    The `!tuple.infer` type represents a *type inference variable* created during
    impl resolution and type checking.

    `!tuple.infer` is identical to `!trait.infer` *except* that it will unify
    only with another tuple-like type.
  }];

  let extraClassDeclaration = [{
    // for PolymorphicTypeInterface
    bool isPolymorphic() const { return true; }

    // for UnificationTypeInterface
    LogicalResult unify(Type other,
                        ModuleOp module,
                        DenseMap<Type,Type> &subst,
                        llvm::function_ref<InFlightDiagnostic()> emitError);
  }];

  let parameters = (ins "uint64_t":$unique_id, "uint32_t":$origin_poly_id);

  let assemblyFormat = "`<` $unique_id `,` $origin_poly_id `>`";
}

#endif // TUPLE_TYPES_TD
