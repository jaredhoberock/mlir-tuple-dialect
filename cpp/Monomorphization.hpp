#pragma once

namespace mlir {

class RewritePatternSet;
class TypeConverter;

namespace tuple {

void populateConvertTupleToTraitPatterns(RewritePatternSet& patterns);
void populateInstantiateMonomorphsPatterns(RewritePatternSet& patterns);
void populateEraseClaimsPatterns(TypeConverter& converter, RewritePatternSet& patterns);

}
}
