#ifndef TUPLE_OPS_TD
#define TUPLE_OPS_TD

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "Tuple.td"
include "TupleEnums.td"

def TupleLike : TypeConstraint<
  CPred<"::mlir::tuple::isTupleLike($_self)">,
  "either tuple<...> or !tuple.poly">;

def ConcreteTuple : TypeConstraint<
  CPred<"::mlir::TupleType::classof($_self)">,
  "any concrete tuple type">;

def AllOp : TupleOp<"all", [
  AutomaticAllocationScope,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  OpAsmOpInterface
]> {
  let summary = "Logical AND of a per-element predicate over a tuple";
  let description = [{
    Executes a region once per element of the input tuple; the region takes the
    current element as a single argument and must yield an `i1`. The op returns
    the logical AND of all yielded bits. For `tuple<>`, the result is `true`.
  }];

  let arguments = (ins TupleLike:$input);
  let results = (outs I1:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$input attr-dict `:` type($input) $body";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    YieldOp bodyYield();

    FunctionType getBodyFunctionType();
    FunctionType getFunctionTypeForIteration(unsigned int i);

    // this method succeeds only when the input tuple is a concrete TupleType
    // element types of the input tuple are still allowed to be polymorphic
    FailureOr<TupleType> getInputTupleTypeWithKnownArity() {
      if (auto tt = dyn_cast<TupleType>(getInput().getType())) {
        return success(tt);
      }
      return failure();
    }

    std::optional<unsigned> getArity() {
      auto maybeTupleOfKnownArity = getInputTupleTypeWithKnownArity();
      if (succeeded(maybeTupleOfKnownArity)) return maybeTupleOfKnownArity->size();
      return std::nullopt;
    }

    LogicalResult verifySymbolUsesWithKnownArity(ModuleOp module, unsigned arity);
    LogicalResult verifySymbolUsesWithUnknownArity(ModuleOp module);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "tuple"; }
  }];
}

def AppendOp : TupleOp<"append", [Pure]> {
  let summary = "Append a element to the end of a tuple";
  let description = [{
    Creates a new `tuple` value from an existing `tuple` and element to append.
  }];

  let arguments = (ins TupleLike:$tuple, AnyType:$element);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$tuple `,` $element `:` type($tuple) `,` type($element) `->` type($result) attr-dict";

  let hasVerifier = 1;
}

def CatOp : TupleOp<"cat", [Pure]> {
  let summary = "Concatenate two tuples (binary cat)";
  let description = [{
    Produces a tuple whose elements are the elements of `lhs` followed by the
    elements of `rhs`.
  }];

  let arguments = (ins TupleLike:$lhs, TupleLike:$rhs);
  let results   = (outs TupleLike:$result);

  let assemblyFormat = "$lhs `,` $rhs `:` type($lhs) `,` type($rhs) `->` type($result) attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static FailureOr<Type> inferResultType(Type lhsTy, Type rhsTy, llvm::function_ref<InFlightDiagnostic()> errFn = nullptr);
  }];

  let builders = [
    // infers the result type
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      auto maybeResTy = CatOp::inferResultType(lhs.getType(), rhs.getType());
      Type resTy = succeeded(maybeResTy) ? *maybeResTy : Type();
      build($_builder, $_state, resTy, lhs, rhs);
    }]>
  ];
}

def CmpOp : TupleOp<"cmp", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Compare two `tuple` values.";
  let description = [{
    Compares two tuple values using a comparison predicate.
  }];

  let arguments = (ins
    CmpPredicateAttr:$predicate,
    TupleLike:$lhs,
    TupleLike:$rhs,
    Optional<AnyType>:$claims
  );

  let results = (outs I1);

  let builders = [
    // with claims
    OpBuilder<(ins "CmpPredicate":$predicate, "Value":$lhs, "Value":$rhs, "Value":$claims), [{
      build($_builder, $_state,
            $_builder.getI1Type(),
            $_builder.getAttr<::mlir::tuple::CmpPredicateAttr>(predicate),
            lhs, rhs, claims);
    }]>,

    // without claims
    OpBuilder<(ins "CmpPredicate":$predicate, "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state,
            predicate, lhs, rhs,
            /*claims=*/Value{});
    }]>
  ];

  let extraClassDeclaration = [{
    FailureOr<std::optional<unsigned>> verifyArity(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    std::optional<unsigned> getArity() {
      auto maybeArity = verifyArity();
      if (succeeded(maybeArity)) return *maybeArity;
      return std::nullopt;
    }

    bool hasClaimsOperand() { return (bool)getClaims(); }

    FailureOr<std::pair<TupleType,TupleType>> getMonomorphicTupleOperandTypes() {
      TupleType L = dyn_cast<TupleType>(getLhs().getType());
      TupleType R = dyn_cast<TupleType>(getRhs().getType());
      if (!L || !R)
        return failure();
      if (!trait::isMonomorphicType(L) || !trait::isMonomorphicType(R))
        return failure();
      return std::make_pair(L,R);
    }

    FailureOr<mlir::trait::TraitOp> getTrait(llvm::function_ref<InFlightDiagnostic()> err = nullptr);
    FlatSymbolRefAttr getTraitRefAttr();
    StringRef getTraitName();
    StringRef getMethodName();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def FlattenOp : TupleOp<"flatten", [Pure]> {
  let summary = "One-level flattening of a tuple of tuples";
  let description = [{
    `tuple.flatten` performs a shallow flatten of a single tuple of tuples.
    The result is the concatenation of all the outer tuple elements.
  }];

  let arguments = (ins TupleLike:$input);
  let results = (outs TupleLike:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

// XXX TODO tuple.flat_map should probably be variadic like tuple.map
def FlatMapOp : TupleOp<"flat_map", [
  AutomaticAllocationScope,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  OpAsmOpInterface
]> {
  let summary = "Map each element to a tuple and concatenate the results";
  let description = [{
    `tuple.flat_map` executes a region once per element of the input tuple.
    The region takes the current element as its single block argument and must
    yield a tuple-like value. The result is the concatenation of all yielded
    tuples in left-to-right order. For `tuple<>`, the result is `tuple<>`.
  }];

  let arguments = (ins TupleLike:$input);
  let results = (outs TupleLike:$result);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result) $body";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    YieldOp bodyYield();

    FunctionType getBodyFunctionType();
    FunctionType getFunctionTypeForIteration(unsigned int i);

    // this method succeeds only when the input tuple is a concrete TupleType
    // element types of the input tuple are still allowed to be polymorphic
    FailureOr<TupleType> getInputTupleTypeWithKnownArity() {
      if (auto tt = dyn_cast<TupleType>(getInput().getType()))
        return success(tt);
      return failure();
    }

    std::optional<unsigned> getArity() {
      auto maybeTupleOfKnownArity = getInputTupleTypeWithKnownArity();
      if (succeeded(maybeTupleOfKnownArity)) return maybeTupleOfKnownArity->size();
      return std::nullopt;
    }

    LogicalResult verifySymbolUsesWithKnownArity(ModuleOp module, unsigned arity);
    LogicalResult verifySymbolUsesWithUnknownArity(ModuleOp module);

    FailureOr<DenseMap<Type,Type>> buildSubstitutionForIteration(
        unsigned int i,
        function_ref<InFlightDiagnostic()> errFn = nullptr);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "tuple"; }
  }];
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
    FailureOr<std::optional<unsigned>> verifyArity(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    std::optional<unsigned> getArity() {
      auto maybeArity = verifyArity();
      if (succeeded(maybeArity)) return *maybeArity;
      return std::nullopt;
    }

    LogicalResult verifySymbolUsesWithKnownArity(ModuleOp module, unsigned arity);
    LogicalResult verifySymbolUsesWithUnknownArity(ModuleOp module);

    llvm::DenseMap<Type,Type> buildSubstitutionForIteration(unsigned int i, Type resultTypeOfPreviousIteration);

    // this method succeeds only when each input tuple is a concrete TupleType
    // the element types of each input tuple are still allowed to be polymorphic
    FailureOr<SmallVector<TupleType>> getInputTypesAsTupleTypes() {
      SmallVector<TupleType> tupleTypes;
      tupleTypes.reserve(getInputs().size());
      for (Value v : getInputs()) {
        if (auto tt = dyn_cast<TupleType>(v.getType())) {
          tupleTypes.push_back(tt);
        } else {
          return failure();
        }
      }
      return tupleTypes;
    }

    bool inputTypesAreTupleTypes() {
      return succeeded(getInputTypesAsTupleTypes());
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
    FailureOr<std::optional<unsigned>> verifyArity(llvm::function_ref<InFlightDiagnostic()> err = nullptr);

    std::optional<unsigned> getArity() {
      auto maybeArity = verifyArity();
      if (succeeded(maybeArity)) return *maybeArity;
      return std::nullopt;
    }

    LogicalResult verifySymbolUsesWithKnownArity(ModuleOp module, unsigned arity);
    LogicalResult verifySymbolUsesWithUnknownArity();

    FailureOr<DenseMap<Type,Type>> buildSubstitutionForIteration(
        unsigned int i,
        function_ref<InFlightDiagnostic()> errFn = nullptr);

    // this method succeeds only when each input tuple is a concrete TupleType
    // the element types of each input tuple are still allowed to be polymorphic
    FailureOr<SmallVector<TupleType>> getInputTypesAsTupleTypes() {
      SmallVector<TupleType> tupleTypes;
      tupleTypes.reserve(getInputs().size());
      for (Value v : getInputs()) {
        if (auto tt = dyn_cast<TupleType>(v.getType())) {
          tupleTypes.push_back(tt);
        } else {
          return failure();
        }
      }
      return tupleTypes;
    }

    bool inputTypesAreTupleTypes() {
      return succeeded(getInputTypesAsTupleTypes());
    }

    // returns the concrete TupleType of the result
    // only valid in the known-arity case
    TupleType getResultTupleType() {
      assert(getArity() && "getResultTupleType called when arity is unknown");
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
  ParentOneOf<["AllOp","FlatMapOp","FoldlOp","MapOp"]>,
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
