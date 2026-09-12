// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "ConvertToLLVM.hpp"
#include "Tuple.hpp"
#include "TupleTypes.hpp"
#include <atomic>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
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

/// The LLVM struct a tuple lowers to, recovered through the lowering's own type
/// conversion. Building a fresh converter and populating it with the tuple
/// mapping means a nested tuple converts exactly as the lowering converts it,
/// so this is byte for byte the struct `--convert-tuple-to-llvm` would produce.
///
/// The converter loads the LLVM dialect as it is built. A tuple's layout is
/// only queried once its IR is on the way to LLVM, where that dialect is
/// already loaded, so the build never first-loads a dialect mid-pass.
static Type loweredStruct(Type type) {
  LLVMTypeConverter converter(type.getContext());
  populateTupleToLLVMTypeConversions(converter);
  return converter.convertType(type);
}

/// Answers size and alignment queries for the builtin tuple type.
///
/// A tuple carries no layout of its own; it lowers to an LLVM literal struct,
/// so every query forwards to the data layout over that struct. The struct
/// comes from the lowering's own type conversion [loweredStruct], so the layout
/// a tuple reports is the layout of the exact struct it lowers to -- nested
/// tuples and the empty tuple's i8 included, because the shared conversion
/// already handles both.
///
/// A builtin type carries no baked-in layout methods, so this model provides
/// every query directly rather than relying on the interface's defaults, which
/// would dispatch back to methods the builtin type does not have.
struct TupleDataLayout
    : public DataLayoutTypeInterface::ExternalModel<TupleDataLayout, TupleType> {
  llvm::TypeSize getTypeSizeInBits(Type type, const DataLayout& dataLayout,
                                   DataLayoutEntryListRef) const {
    return dataLayout.getTypeSizeInBits(loweredStruct(type));
  }

  llvm::TypeSize getTypeSize(Type type, const DataLayout& dataLayout,
                             DataLayoutEntryListRef) const {
    return dataLayout.getTypeSize(loweredStruct(type));
  }

  uint64_t getABIAlignment(Type type, const DataLayout& dataLayout,
                           DataLayoutEntryListRef) const {
    return dataLayout.getTypeABIAlignment(loweredStruct(type));
  }

  uint64_t getPreferredAlignment(Type type, const DataLayout& dataLayout,
                                 DataLayoutEntryListRef) const {
    return dataLayout.getTypePreferredAlignment(loweredStruct(type));
  }
};

} // namespace

void registerTupleDataLayoutInterface(MLIRContext* ctx) {
  TupleType::attachInterface<TupleDataLayout>(*ctx);
}


//===----------------------------------------------------------------------===//
// PolyType
//===----------------------------------------------------------------------===//

PolyType PolyType::getUnique(MLIRContext* ctx) {
  trait::PolyType inner = trait::PolyType::getUnique(ctx);
  return PolyType::get(ctx, inner);
}

// XXX TODO A parameter stands for itself; nothing mints a variable for one any
// more. Deleted when `GenericTypeInterface` loses `instantiate`.
Type PolyType::instantiate(trait::InstantiationMap &, uint64_t &) {
  return *this;
}

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
  // !tuple.poly<unique>
  // !tuple.poly<N>
  // !tuple.poly<!trait.poly<N>>

  MLIRContext *ctx = parser.getContext();
  
  if (parser.parseLess())
    return {};

  trait::PolyType inner;
  if (succeeded(parser.parseOptionalKeyword("unique"))) {
    inner = trait::PolyType::getUnique(ctx);
  } else {
    int uniqueId;
    auto intResult = parser.parseOptionalInteger(uniqueId);
    if (intResult.has_value() && succeeded(*intResult)) {
      inner = trait::PolyType::get(ctx, uniqueId);
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
  }

  if (parser.parseGreater())
    return {};

  return PolyType::get(ctx, inner);
}

void PolyType::print(AsmPrinter &printer) const {
  printer << "<" << getInner().getUniqueId() << ">";
}

} // end mlir::tuple
