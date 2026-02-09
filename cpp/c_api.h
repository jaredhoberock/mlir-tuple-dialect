// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

void tupleRegisterDialect(MlirContext ctx);

MlirOperation tupleMakeOpCreate(MlirLocation loc, MlirValue* elements, intptr_t nElements);

MlirOperation tupleGetOpCreate(MlirLocation loc, MlirValue tuple, int64_t index);

typedef enum {
  TupleCmpPredicateEq = 0,
  TupleCmpPredicateNe,
  TupleCmpPredicateLt,
  TupleCmpPredicateLe,
  TupleCmpPredicateGt,
  TupleCmpPredicateGe
} TupleCmpPredicate;

// If `claims` is null (mlirValueIsNull), the op is created without a claims operand.
MlirOperation tupleCmpOpCreate(MlirLocation loc, 
                               TupleCmpPredicate predicate,
                               MlirValue lhs,
                               MlirValue rhs,
                               MlirValue claims);
      

#ifdef __cplusplus
}
#endif
