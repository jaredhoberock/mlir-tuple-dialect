// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#ifndef TUPLE_CONSTRAINTS_TD
#define TUPLE_CONSTRAINTS_TD

include "mlir/IR/OpBase.td"

// A type is tuple-like when it is a concrete tuple<...> or the polymorphic
// !tuple.poly placeholder. Downstream dialects include this constraint rather
// than restating the predicate.
def TupleLike : TypeConstraint<
  CPred<"::mlir::tuple::isTupleLike($_self)">,
  "either tuple<...> or !tuple.poly">;

#endif // TUPLE_CONSTRAINTS_TD
