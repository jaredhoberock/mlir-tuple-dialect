// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "Tuple.hpp"
#include "TupleTypes.hpp"
#include <atomic>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
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
/// conversion. The converter is populated from every loaded dialect's
/// ConvertToLLVMPatternInterface, the population upstream's convert-to-llvm
/// performs in its normal mode, so a nested type converts exactly as the
/// lowering converts it: a nested tuple through the tuple mapping, and any other
/// element -- a sum, say -- through its own dialect's conversion. The target and
/// pattern set are throwaway; only the type conversions the interfaces add to
/// the converter are consulted.
///
/// The converter loads the LLVM dialect as it is built. A tuple's layout is
/// only queried once its IR is on the way to LLVM, where that dialect is
/// already loaded, so the build never first-loads a dialect mid-pass.
///
/// A tuple holding an element no loaded dialect lowers has no LLVM struct and
/// so no layout. That stops compilation with a fatal error naming the tuple, as
/// the data layout itself stops on a type that provides no layout, instead of
/// handing a null type to the layout query.
static Type loweredStruct(Type type) {
  MLIRContext *ctx = type.getContext();
  LLVMTypeConverter converter(ctx);
  ConversionTarget throwawayTarget(*ctx);
  RewritePatternSet throwawayPatterns(ctx);
  for (Dialect *dialect : ctx->getLoadedDialects())
    if (auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect))
      iface->populateConvertToLLVMConversionPatterns(throwawayTarget, converter,
                                                     throwawayPatterns);

  Type lowered = converter.convertType(type);
  if (!lowered) {
    std::string message;
    llvm::raw_string_ostream os(message);
    os << "no loaded dialect lowers every element of " << type
       << " to LLVM, so the tuple has no data layout";
    llvm::report_fatal_error(llvm::Twine(message));
  }
  return lowered;
}

/// Answers size and alignment queries for the builtin tuple type.
///
/// A tuple carries no layout of its own; it lowers to an LLVM literal struct,
/// so every query forwards to the data layout over that struct. The struct
/// comes from the lowering's own type conversion [loweredStruct], so the layout
/// a tuple reports is the layout of the exact struct it lowers to -- nested
/// tuples, elements of other dialects, and the empty tuple's i8 included,
/// because every loaded dialect's conversion takes part, as in the lowering.
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
