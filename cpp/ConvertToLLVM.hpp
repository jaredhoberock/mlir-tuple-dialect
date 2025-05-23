#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace tuple {

void populateTupleToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                           RewritePatternSet& patterns);
}
}

