#pragma once

namespace mlir {

class RewritePatternSet;

namespace tuple {

void populateTupleToTraitConversionPatterns(RewritePatternSet& patterns);
}
}
