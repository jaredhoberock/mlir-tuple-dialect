// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace mlir {

namespace trait {

class ImplGeneratorSet;

}

namespace tuple {

void populateImplGenerators(trait::ImplGeneratorSet& generators);

}
}
