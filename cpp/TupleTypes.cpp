#include "Tuple.hpp"
#include "TupleTypes.hpp"
#include <atomic>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <TraitOps.hpp>
#include <TraitTypes.hpp>

#define GET_TYPEDEF_CLASSES
#include "TupleTypes.cpp.inc"

namespace mlir::tuple {


void TupleDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "TupleTypes.cpp.inc"
  >();
}


//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

static int nextPolyTypeId() {
  static std::atomic<int> counter{-1};
  return counter.fetch_sub(1, std::memory_order_relaxed);
}

PolyType PolyType::getUnique(MLIRContext* ctx) {
  return PolyType::get(ctx, nextPolyTypeId());
}

Type PolyType::instantiate(DenseMap<Type,Type> &inst, uint64_t &idCounter) {
  // check memo first - if we've already instantiated this PolyType,
  // return the instance
  if (auto it = inst.find(*this); it != inst.end()) {
    return it->second;
  }

  // create and remember a fresh inference var for this poly
  auto fresh = InferenceType::get(getContext(), idCounter++, getUniqueId());
  inst[*this] = fresh;
  return fresh;
}

Type PolyType::parse(AsmParser &parser) {
  MLIRContext *ctx = parser.getContext();
  int uniqueId = 0;

  // parse this:
  // <unique> or
  // <int>

  if (parser.parseLess()) {
    parser.emitError(parser.getNameLoc(), "expected '<'");
    return Type();
  }

  if (succeeded(parser.parseOptionalKeyword("unique"))) {
    uniqueId = nextPolyTypeId();
  } else {
    if (parser.parseInteger(uniqueId)) {
      parser.emitError(parser.getNameLoc(), "expected integer or 'unique'");
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


//===----------------------------------------------------------------------===//
// InferenceType
//===----------------------------------------------------------------------===//

LogicalResult InferenceType::unify(
  Type other,
  ModuleOp /*module*/,
  DenseMap<Type,Type> &subst,
  llvm::function_ref<InFlightDiagnostic()> err) {
  Type self = *this;

  // normalize
  other = trait::applySubstitution(subst, other);

  // first check for trivial equality
  if (self == other) return success();

  // if self is already bound, check consistency
  if (auto it = subst.find(self); it != subst.end()) {
    if (it->second != other) {
      if (err) return err() << "inference variable " << self
                            << " already bound to " << it->second
                            << ", cannot bind to " << other;
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

  // accept only tuple-like types
  if (isTupleLike(other)) {
    subst[self] = other;
    return success();
  }

  // otherwise, reject
  if (err) err() << "type mismatch: expected a tuple type, but found " << other;
  return failure();
}

} // end mlir::tuple
