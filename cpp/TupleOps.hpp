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
#include "TupleOps.hpp.inc"
