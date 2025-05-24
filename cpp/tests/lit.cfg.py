import os
import lit.formats

config.name = "Tuple Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

tuple_plugin_path = os.path.join(os.path.dirname(__file__), '..', 'libtuple_dialect.so')
trait_plugin_path = os.path.join(os.path.dirname(__file__), '../../../mlir-trait-dialect/cpp', 'libtrait_dialect.so')

config.substitutions.append(('mlir-opt', f'mlir-opt --load-dialect-plugin={trait_plugin_path} --load-dialect-plugin={tuple_plugin_path}'))
