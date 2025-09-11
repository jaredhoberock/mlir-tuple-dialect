#ifndef TUPLE_TYPES_TD
#define TUPLE_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "Tuple.td"
include "TraitTypeInterfaces.td"

def PolyType : TypeDef<TupleDialect, "Poly", [
  MonomorphizableTypeInterface,
  TypeVariableInterface
]> {
  let mnemonic = "poly";
  let summary = "Placeholder type representing a polymorphic tuple type";
  let description = [{
    The `!tuple.poly` type represents a universally quantified type parameter
    that is also a tuple. It is used to model the parameters of generic functions
    on tuples.

    The `unique_id` serves as a unique identifier for this type within the module.

    Alternatively, `fresh` can be specified instead of `unique_id` to request a "fresh"
    `!tuple.poly` with an automatically-generated `unique_id`. Fresh IDs are allocated in
    monotonically decreasing order beginning at `-1`.
  }];

  let extraClassDeclaration = [{
    // create a "fresh" PolyType with an unused unique_id
    static PolyType fresh(MLIRContext* ctx);

    // for MonomorphizableTypeInterface
    bool isMonomorphic() const {
      return false;
    }

    LogicalResult substituteWith(Type other,
                                 ModuleOp module,
                                 DenseMap<Type,Type> &substitution,
                                 llvm::function_ref<InFlightDiagnostic()> emitError) const;
  }];

  let parameters = (ins "int":$unique_id);

  let hasCustomAssemblyFormat = 1;
}

#endif // TUPLE_TYPES_TD
