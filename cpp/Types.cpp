#include "Dialect.hpp"
#include "Types.hpp"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <TraitOps.hpp>

#define GET_TYPEDEF_CLASSES
#include "Types.cpp.inc"

namespace mlir::tuple {


void TupleDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Types.cpp.inc"
  >();
}


//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

PolyType PolyType::fresh(MLIRContext* ctx) {
  return PolyType::get(ctx, trait::freshPolyTypeId());
}

Type PolyType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  int uniqueId = 0;

  // parse this:
  // <fresh> or
  // <int>

  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expected '<'");
    return Type();
  }

  if (succeeded(parser.parseOptionalKeyword("fresh"))) {
    uniqueId = trait::freshPolyTypeId();
  } else {
    if (parser.parseInteger(uniqueId)) {
      parser.emitError(parser.getNameLoc(), "expected integer or 'fresh'");
      return Type();
    }
    
  }

  if (parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "expected '>'");
    return Type();
  }

  return PolyType::get(ctx, uniqueId);
}

void PolyType::print(AsmPrinter &printer) const {
  printer << "<" << getUniqueId() << ">";
}

LogicalResult PolyType::substituteWith(
  Type other, 
  ModuleOp /*module*/,
  DenseMap<Type,Type> &subst,
  llvm::function_ref<InFlightDiagnostic()> err) const {
  Type self = *this;

  // normalize
  other = trait::applySubstitution(subst, other);

  // first check for trivial equality
  if (self == other) return success();

  // if self is already bound, check consistency
  if (auto it = subst.find(self); it != subst.end()) {
    if (it->second != other) {
      if (err) return err() << "mismatched substitution for type "
                            << self << ": expected "
                            << it->second << ", but found " << other;
      return failure();
    }
    return success();
  }

  // occurs check: forbid T := f(..., T, ...) to avoid cycles
  auto occursIn = [](Type needle, Type haystack) {
    bool hit = false;
    haystack.walk([&](Type t) {
      if (!hit && t == needle) hit = true;
    });
    return hit;
  };

  if (occursIn(self, other)) {
    if (err) err() << "recursive substitution: " << self
                   << " occurs in " << other;
    return failure();
  }

  // accept either PolyType or a concrete TupleType
  if (mlir::isa<PolyType>(other) || mlir::isa<TupleType>(other)) {
    subst[self] = other;
    return success();
  }

  // otherwise, reject
  if (err) err() << "type mismatch: expected a tuple type, but found " << other;
  return failure();
}

} // end mlir::tuple
