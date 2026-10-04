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

/// Whether `mapper` maps the trait named `mapped` across the elements of two
/// tuples: it is tagged `tuple.impl_generator = "map"` and its
/// `tuple.mapped_trait` names `mapped`. A mapper is identified by that link,
/// never by its own name, which is a label.
inline bool isMapperOf(trait::TraitOp mapper, FlatSymbolRefAttr mapped) {
  auto tag = mapper->getAttrOfType<StringAttr>("tuple.impl_generator");
  return tag && tag.getValue() == "map" &&
         mapper->getAttrOfType<FlatSymbolRefAttr>("tuple.mapped_trait") == mapped;
}

} // end mlir::tuple
