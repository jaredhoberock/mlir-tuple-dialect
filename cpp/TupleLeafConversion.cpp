// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "TupleLeafConversion.hpp"
#include "TupleOps.hpp"
#include <LeafConversion.hpp>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tuple {

void populateTupleLeafTypes(TypeConverter &converter) {
  converter.addConversion(
      [&converter](TupleType type, SmallVectorImpl<Type> &leaves)
          -> std::optional<LogicalResult> {
        // A refusal leaves `leaves` as it found them, the converter's contract.
        SmallVector<Type> elementLeaves;
        if (failed(converter.convertTypes(type.getTypes(), elementLeaves)))
          return failure();
        llvm::append_range(leaves, elementLeaves);
        return success();
      });

  converter.addSourceMaterialization(
      [&converter](OpBuilder &builder, TupleType type, ValueRange leaves,
                   Location loc) -> Value {
        FailureOr<SmallVector<Value>> elements = lowering::valuesFromLeaves(
            converter, builder, loc, type.getTypes(), leaves);
        if (failed(elements))
          return Value();
        return MakeOp::create(builder, loc, type, *elements);
      });

  converter.addTargetMaterialization(
      [&converter](OpBuilder &builder, TypeRange, ValueRange inputs,
                   Location loc, Type) -> SmallVector<Value> {
        if (inputs.size() != 1 || !isa<TupleType>(inputs.front().getType()))
          return {};
        Value tuple = inputs.front();
        SmallVector<Value> leaves;
        for (unsigned index = 0, e = cast<TupleType>(tuple.getType()).size();
             index != e; ++index) {
          Value element =
              GetOp::create(builder, loc, tuple, static_cast<int64_t>(index));
          if (failed(lowering::appendLeaves(converter, builder, loc, element,
                                            leaves)))
            return {};
        }
        return leaves;
      });
}

} // namespace mlir::tuple
