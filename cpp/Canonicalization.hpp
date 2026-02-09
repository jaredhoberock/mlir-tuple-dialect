// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace mlir {

class RewritePatternSet;

namespace tuple {

void populateTupleCanonicalizationPatterns(RewritePatternSet& patterns);

} // end tuple
} // end mlir
