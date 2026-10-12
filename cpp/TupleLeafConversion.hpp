// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace mlir {

class TypeConverter;

namespace tuple {

/// Adds the builtin tuple type to the leaf conversion `converter`: a tuple
/// converts to the leaves of its elements in order, so `tuple<>` converts to
/// none; its source materialization rebuilds each element from its leaves and
/// packs them with `tuple.make`, and its target materialization reads each
/// element with `tuple.get` and takes it apart in turn. Elements convert and
/// materialize through `converter`, whatever dialect owns them.
void populateTupleLeafTypes(TypeConverter &converter);

} // namespace tuple
} // namespace mlir
