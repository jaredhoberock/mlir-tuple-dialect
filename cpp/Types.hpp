#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir-trait-dialect/cpp/Ops.hpp>
#include <mlir-trait-dialect/cpp/Types.hpp>
#include <optional>

#define GET_TYPEDEF_CLASSES
#include "Types.hpp.inc"

namespace mlir::tuple {

// checks that all the element types of tuple_ty have an impl for the given trait
// it not, returns the Type of the first failing element
std::optional<Type> firstElementTypeWithoutImplForTrait(TupleType tuple_ty, mlir::trait::TraitOp traitOp);

} // end mlir::trait
