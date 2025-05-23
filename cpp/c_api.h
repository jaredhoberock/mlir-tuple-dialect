#pragma once

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

void tupleRegisterDialect(MlirContext ctx);

MlirOperation tupleConstantOpCreate(MlirLocation loc, MlirValue* elements, intptr_t nElements);

MlirOperation tupleGetOpCreate(MlirLocation loc, MlirValue tuple, int64_t index);

typedef enum {
  TupleCmpPredicateEq = 0,
  TupleCmpPredicateNe,
  TupleCmpPredicateLt,
  TupleCmpPredicateLe,
  TupleCmpPredicateGt,
  TupleCmpPredicateGe
} TupleCmpPredicate;

MlirOperation tupleCmpOpCreate(MlirLocation loc, 
                               TupleCmpPredicate predicate,
                               MlirValue lhs,
                               MlirValue rhs);
      

#ifdef __cplusplus
}
#endif
