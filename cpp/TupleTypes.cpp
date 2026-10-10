// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Tuple.hpp"
#include "TupleTypes.hpp"
#include <atomic>
#include <InterfaceLLVMTypeConverter.hpp>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <TraitOps.hpp>
#include <TraitTypes.hpp>

#define GET_TYPEDEF_CLASSES
#include <TupleTypes.cpp.inc>

namespace mlir::tuple {


void TupleDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <TupleTypes.cpp.inc>
  >();
}


//===----------------------------------------------------------------------===//
// Tuple data layout
//===----------------------------------------------------------------------===//

namespace {

/// Answers size and alignment queries for the builtin tuple type.
///
/// A tuple carries no layout of its own; it lowers to an LLVM literal struct,
/// so every query forwards to the data layout over that struct. The struct
/// comes from the lowering's own type conversion
/// [lowering::loweredForDataLayout], so the layout a tuple reports is the layout
/// of the exact struct it lowers to -- nested tuples, elements of other
/// dialects, and the empty tuple's i8 included, because every loaded dialect's
/// conversion takes part, as in the lowering. A tuple holding an element no
/// loaded dialect lowers has no struct and stops compilation with a fatal error
/// naming it.
///
/// A builtin type carries no baked-in layout methods, so this model provides
/// every query directly rather than relying on the interface's defaults, which
/// would dispatch back to methods the builtin type does not have.
struct TupleDataLayout
    : public DataLayoutTypeInterface::ExternalModel<TupleDataLayout, TupleType> {
  llvm::TypeSize getTypeSizeInBits(Type type, const DataLayout& dataLayout,
                                   DataLayoutEntryListRef) const {
    return dataLayout.getTypeSizeInBits(
        lowering::loweredForDataLayout(type, dataLayout));
  }

  llvm::TypeSize getTypeSize(Type type, const DataLayout& dataLayout,
                             DataLayoutEntryListRef) const {
    return dataLayout.getTypeSize(
        lowering::loweredForDataLayout(type, dataLayout));
  }

  uint64_t getABIAlignment(Type type, const DataLayout& dataLayout,
                           DataLayoutEntryListRef) const {
    return dataLayout.getTypeABIAlignment(
        lowering::loweredForDataLayout(type, dataLayout));
  }

  uint64_t getPreferredAlignment(Type type, const DataLayout& dataLayout,
                                 DataLayoutEntryListRef) const {
    return dataLayout.getTypePreferredAlignment(
        lowering::loweredForDataLayout(type, dataLayout));
  }
};

} // namespace

void registerTupleDataLayoutInterface(MLIRContext* ctx) {
  TupleType::attachInterface<TupleDataLayout>(*ctx);
}


//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

Type PolyType::specializeWith(const trait::SpecializationMap &subst) const {
  // specialize the parameter this type constrains
  trait::PolyType inner = getInner();
  Type specialized = inner.specializeWith(subst);
  if (specialized == inner)
    return *this;

  // an argument that is still polymorphic carries the constraint with it
  if (auto poly = llvm::dyn_cast<trait::PolyType>(specialized))
    return PolyType::get(getContext(), poly);

  // This type spells a tuple and nothing else, so an argument that is not one
  // has no spelling here and this type yields none: the constraint stands as
  // written, and whatever compares the rebuilt spelling refuses the argument.
  if (!isTupleLike(specialized))
    return {};

  return specialized;
}

Type PolyType::parse(AsmParser &parser) {
  // These spellings are valid:
  // !tuple.poly<N>
  // !tuple.poly<!trait.poly<N>>

  MLIRContext *ctx = parser.getContext();

  if (parser.parseLess())
    return {};

  trait::PolyType inner;
  llvm::SMLoc labelLoc = parser.getCurrentLocation();
  int label;
  auto intResult = parser.parseOptionalInteger(label);
  if (intResult.has_value() && succeeded(*intResult)) {
    // A label names a position in the declaration that binds it, so it is
    // non-negative; a negative one names no position.
    if (label < 0) {
      parser.emitError(labelLoc, "a !trait.poly label is non-negative; found ")
          << label;
      return {};
    }
    inner = trait::PolyType::get(ctx, label);
  } else {
    Type innerType;
    if (parser.parseType(innerType))
      return {};
    inner = llvm::dyn_cast<trait::PolyType>(innerType);
    if (!inner) {
      parser.emitError(parser.getCurrentLocation(),
                       "inner type of !tuple.poly must be !trait.poly");
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return PolyType::get(ctx, inner);
}

void PolyType::print(AsmPrinter &printer) const {
  printer << "<" << getInner().getLabel() << ">";
}

} // end mlir::tuple
