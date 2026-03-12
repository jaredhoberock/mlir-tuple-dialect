// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace mlir {

class RewritePatternSet;
class TypeConverter;

namespace tuple {

void populateConvertTupleToTraitPatterns(RewritePatternSet& patterns);
void populateInstantiateMonomorphsPatterns(RewritePatternSet& patterns);
void populateErasePolymorphsPatterns(TypeConverter& converter, RewritePatternSet& patterns);

}
}
