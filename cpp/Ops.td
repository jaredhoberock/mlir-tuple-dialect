#ifndef TUPLE_OPS_TD
#define TUPLE_OPS_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "Dialect.td"
include "Enums.td"

def Tuple : TypeConstraint<
  CPred<"::mlir::tuple::isTuple($_self)">,
  "any tuple type">;

def ConcreteTuple : TypeConstraint<
  CPred<"::mlir::TupleType::classof($_self)">,
  "any concrete tuple type">;

def ConstantOp : TupleOp<"constant", [Pure]> {
  let summary = "Construct a constant tuple value";
  let description = [{
    Packs operands into an MLIR `tuple<...>`. Arity and per‑
    element type matching are enforced by the custom verifier.
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyType:$result);

  // optional parens around operands, then a single `: <result-type>`
  let assemblyFormat = "attr-dict (`(` $elements^ `:` type($elements) `)`)? `:` type($result)";

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$elements), [{
      SmallVector<Type> elementTypes;
      elementTypes.reserve(elements.size());
      for (Value e : elements)
        elementTypes.push_back(e.getType());
      auto tupleTy = ::mlir::TupleType::get($_builder.getContext(), elementTypes);
      build($_builder, $_state, tupleTy, elements);
    }]>
  ];
}

def GetOp : TupleOp<"get", [Pure]> {
  let summary = "Extract an element from a tuple";
  let description = [{
    `tuple.get` extracts the element at the given index from the input
    `tuple`. The index must be an integer attribute, and the result type
    must match the type of the element at that index.
  }];

  let arguments = (ins ConcreteTuple:$tuple, IndexAttr:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$tuple `,` $index `:` type($tuple) `->` type($result) attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    TupleType getTupleType() {
      return cast<TupleType>(getOperand().getType());
    }
  }];

  let builders = [
    OpBuilder<(ins "Value":$tuple, "int64_t":$index), [{
      auto tupleTy = cast<TupleType>(tuple.getType());
      auto resultTy = tupleTy.getType(index);
      auto indexAttr = $_builder.getIndexAttr(index);
      build($_builder, $_state, resultTy, tuple, indexAttr);
    }]>
  ];
}

def CmpOp : TupleOp<"cmp", [
  AllTypesMatch<["lhs", "rhs"]>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Compare two `tuple` values.";
  let description = [{
    Compares two tuple values using a comparison predicate.
  }];

  let arguments = (ins
    CmpPredicateAttr:$predicate,
    Tuple:$lhs,
    Tuple:$rhs
  );

  let results = (outs I1);

  let builders = [
    OpBuilder<(ins "CmpPredicate":$predicate, "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state,
            $_builder.getI1Type(),
            $_builder.getAttr<::mlir::tuple::CmpPredicateAttr>(predicate),
            lhs, rhs);
    }]>
  ];

  let extraClassDeclaration = [{
    mlir::trait::TraitOp getTrait();
    mlir::FlatSymbolRefAttr getTraitRefAttr();
    StringRef getTraitName();
    StringRef getMethodName();
    LogicalResult verifyTupleTypeHasImplFor(mlir::trait::TraitOp traitOp, Type tuple_ty);
  }];

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

#endif // TUPLE_OPS_TD
