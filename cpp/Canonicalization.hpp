#pragma once

namespace mlir {

class RewritePatternSet;

namespace tuple {

void populateTupleCanonicalizationPatterns(RewritePatternSet& patterns);

} // end tuple
} // end mlir
