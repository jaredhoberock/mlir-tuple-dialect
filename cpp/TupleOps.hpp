// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "TupleEnums.hpp"
#include "TupleTypes.hpp"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <TraitOps.hpp>

#define GET_OP_CLASSES
#include <TupleOps.hpp.inc>

namespace mlir::tuple {

/// The name of the trait that maps `mappedTraitName` across the elements of
/// two tuples and binds the tuple of the resulting claims as its `Claims`
/// associated type. One spelling: the pattern that introduces the mapper, the
/// generator that implements it, and `tuple.cmp` all name it from here.
inline std::string getMapperTraitName(StringRef mappedTraitName) {
  return (Twine("tuple.Map") + mappedTraitName).str();
}

} // end mlir::tuple
