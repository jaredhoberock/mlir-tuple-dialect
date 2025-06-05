#ifndef TUPLE_TYPES_TD
#define TUPLE_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir-trait-dialect/cpp/TypeInterfaces.td"
include "Dialect.td"

def AnyTupleType : TypeDef<TupleDialect, "AnyTuple", [SymbolicTypeUnificationInterface]> {
  let mnemonic = "any";
  let summary = "Placeholder type representing any tuple type with trait bounds on element types";
  let description = [{
    The `!tuple.any` type is a symbolic type used to provide a generic `trait.impl`s
    for all tuple types.

    It provides the method `unifyWith` which will return `true` for any tuple type
    whose element types all have a `trait.impl` for each trait mentioned in `traitBounds`.

    For example:
      trait.trait @PartialEq {
        func.func private @eq(!trait.self, !trait.self) -> i1
      }

      trait.impl @PartialEq !tuple.any<[@PartialEq]> {
        func.func private @eq(%a: !tuple.any<[@PartialEq]>, %b: !tuple.any<[@PartialEq]>) -> i1 {
          %res = tuple.cmp eq, %a, %b : !tuple.any<[@PartialEq]>
          return $res : i1
        }
      }
  }];

  let extraClassDeclaration = [{
    // for SymbolicTypeUnificationInterface
    mlir::LogicalResult unifyWith(mlir::Type ty, ModuleOp module, llvm::function_ref<InFlightDiagnostic()> emitError) const;

    bool hasTraitBound(::mlir::FlatSymbolRefAttr traitRefAttr) const {
      for (auto attr : getTraitBounds()) {
        if (traitRefAttr == attr) return true;
      }
      return false;
    }

    SmallVector<trait::TraitOp> getTraits(ModuleOp module) const;
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::FlatSymbolRefAttr">:$traitBounds
  );

  let assemblyFormat = "`<` `[` $traitBounds `]` `>`";
}

#endif // TUPLE_TYPES_TD
