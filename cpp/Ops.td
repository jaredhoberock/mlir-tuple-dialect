#ifndef TUPLE_OPS_TD
#define TUPLE_OPS_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
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

def AppendOp : TupleOp<"append", [Pure]> {
  let summary = "Append a element to the end of a tuple";
  let description = [{
    Creates a new `tuple` value from an existing `tuple` and element to append.
  }];

  let arguments = (ins Tuple:$tuple, AnyType:$element);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$tuple `,` $element `:` type($tuple) `,` type($element) `->` type($result) attr-dict";

  let hasVerifier = 1;
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

def FoldlOp : TupleOp<"foldl", [
  AutomaticAllocationScope,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  OpAsmOpInterface
]> {
  let summary = "Accumulate a state left to right across tuple elements";
  let description = [{
    `tuple.foldl` executes a region sequentially on each element of tuple
    while accumulating a state. The final state is returned as the result of
    the `tuple.foldl`.
  }];

  let arguments = (ins AnyType:$init, Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    unsigned int getArity() {
      return getInputTupleTypes()[0].size();
    }

    llvm::DenseMap<Type,Type> buildSubstitutionForIteration(unsigned int i, Type resultTypeOfPreviousIteration);

    SmallVector<TupleType> getInputTupleTypes() {
      SmallVector<TupleType> result;
      for (auto input: getInputs()) {
        result.push_back(cast<TupleType>(input.getType()));
      }
      return result;
    }

    YieldOp bodyYield();

    FunctionType getBodyFunctionType();
    FunctionType getFunctionTypeForIteration(unsigned int i, Type resultTypeOfPreviousIteration);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "tuple"; }
  }];

  let assemblyFormat = "$init `,` $inputs attr-dict `:` type($init) `,` type($inputs) `->` type($result) $body";
  let hasVerifier = 1;
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

def MakeOp : TupleOp<"make", [Pure]> {
  let summary = "Construct a tuple value";
  let description = [{
    Packs operands into a `tuple<...>`.
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

def MapOp : TupleOp<"map", [
  AutomaticAllocationScope,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  OpAsmOpInterface
]> {
  let summary = "Execute a region elementwise across tuples";
  let description = [{
    `tuple.map` executes a region elementwise across multiple input tuples.
    All input tuples must have the same arity. The region contains a single
    block with one argument per tuple element position.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    unsigned int getArity() {
      return getResultTupleType().size();
    }

    llvm::DenseMap<Type,Type> buildSubstitutionForIteration(unsigned int i);

    SmallVector<TupleType> getInputTupleTypes() {
      SmallVector<TupleType> result;
      for (auto input : getInputs()) {
        result.push_back(cast<TupleType>(input.getType()));
      }
      return result;
    }

    TupleType getResultTupleType() {
      return cast<TupleType>(getResult().getType());
    }

    YieldOp bodyYield();

    FunctionType getBodyFunctionType();
    FunctionType getFunctionTypeForIteration(unsigned int i);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "tuple"; }
  }];

  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($result) $body";
  let hasVerifier = 1;
}

def YieldOp : TupleOp<"yield", [
  ParentOneOf<["FoldlOp","MapOp"]>,
  Pure,
  ReturnLike,
  Terminator
]> {
  let summary = "Yield operation for `tuple.map`";
  let description = [{
    `tuple.yield` yields a single value from a region within a tuple operation.
    This is used as the terminator for regions like those in `tuple.map`.
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "attr-dict $result `:` type($result)";
}

#endif // TUPLE_OPS_TD
