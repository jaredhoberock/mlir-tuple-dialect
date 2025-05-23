#pragma once

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir-trait-dialect/cpp/Ops.hpp>
#include "Enums.hpp"
#include "Types.hpp"

#define GET_OP_CLASSES
#include "Ops.hpp.inc"
