#pragma once

#include "Enums.hpp"
#include "Types.hpp"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <TraitOps.hpp>

#define GET_OP_CLASSES
#include "Ops.hpp.inc"
