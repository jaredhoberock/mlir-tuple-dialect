#pragma once

namespace mlir {

namespace trait {

class ImplGeneratorSet;

}

namespace tuple {

void populateImplGenerators(trait::ImplGeneratorSet& generators);

}
}
