#include "Dialect.hpp"
#include <mlir/Tools/Plugins/DialectPlugin.h>

static void registerPlugin(mlir::DialectRegistry* registry) {
  registry->insert<mlir::tuple::TupleDialect>();
}

extern "C" ::mlir::DialectPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
mlirGetDialectPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "TupleDialectPlugin", 
    "v0.1", 
    registerPlugin
  };
}
