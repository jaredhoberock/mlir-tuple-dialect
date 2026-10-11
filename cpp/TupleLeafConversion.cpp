// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#include "TupleLeafConversion.hpp"
#include "TupleOps.hpp"
#include <LeafConversion.hpp>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tuple {

void populateTupleLeafConversion(ConversionTarget &, TypeConverter &converter,
                                 RewritePatternSet &) {
  converter.addConversion(
      [&converter](TupleType type, SmallVectorImpl<Type> &leaves)
          -> std::optional<LogicalResult> {
        for (Type element : type.getTypes())
          if (failed(converter.convertType(element, leaves)))
            return failure();
        return success();
      });

  // Each element takes as many leaves as it converts to, in order, so the
  // leaves are consumed front to back and must be used up exactly.
  converter.addSourceMaterialization(
      [&converter](OpBuilder &builder, TupleType type, ValueRange leaves,
                   Location loc) -> Value {
        SmallVector<Value> elements;
        for (Type element : type.getTypes()) {
          SmallVector<Type> elementLeaves;
          if (failed(converter.convertType(element, elementLeaves)) ||
              elementLeaves.size() > leaves.size())
            return Value();
          Value rebuilt = lowering::valueFromLeaves(
              converter, builder, loc, element,
              leaves.take_front(elementLeaves.size()));
          if (!rebuilt)
            return Value();
          elements.push_back(rebuilt);
          leaves = leaves.drop_front(elementLeaves.size());
        }
        if (!leaves.empty())
          return Value();
        return MakeOp::create(builder, loc, type, elements);
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
