#ifndef TUPLE_DIALECT_TD
#define TUPLE_DIALECT_TD

include "mlir/IR/OpBase.td"

def TupleDialect : Dialect {
  let name = "tuple";
  let summary = "A dialect for operations on tuples";
  let cppNamespace = "::mlir::tuple";

  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let dependentDialects = [
    "mlir::trait::TraitDialect",
    "mlir::vector::VectorDialect"
  ];

  let hasCanonicalizer = 1;
}

class TupleOp<string mnemonic, list<Trait> traits = []> :
  Op<TupleDialect, mnemonic, traits>;

#endif // TUPLE_DIALECT_TD
