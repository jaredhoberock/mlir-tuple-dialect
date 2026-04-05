# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
LLVM_BIN_PATH = $(HOME)/dev/git/llvm-project-22/build/bin

LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen
OPT := $(LLVM_BIN_PATH)/mlir-opt

# Compiler flags
CXX := clang++
CXXFLAGS := -Wall -Werror -g -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
MLIR_INCLUDE = $(HOME)/dev/git/llvm-project-22/install/include

# Trait dialect include
TRAIT_DIALECT_INCLUDE = $(HOME)/dev/git/mlir-trait-dialect/cpp

INCLUDES := -I $(MLIR_INCLUDE) -I $(TRAIT_DIALECT_INCLUDE)

# Dialect library sources
DIALECT_SOURCES := c_api.cpp Canonicalization.cpp ConvertToLLVM.cpp ImplGenerators.cpp Monomorphization.cpp Tuple.cpp TupleEnums.cpp TupleOps.cpp TupleTypes.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Tuple.hpp.inc Tuple.cpp.inc TupleEnums.hpp.inc TupleEnums.cpp.inc TupleOps.hpp.inc TupleOps.cpp.inc TupleTypes.hpp.inc TupleTypes.cpp.inc

.PHONY: all clean

all: libtuple_dialect.a libtuple_dialect.so

# TableGen rules
Tuple.hpp.inc: Tuple.td
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

Tuple.cpp.inc: Tuple.td
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

TupleEnums.hpp.inc: TupleEnums.td
	$(TBLGEN) --gen-enum-decls $(INCLUDES) $< -o $@

TupleEnums.cpp.inc: TupleEnums.td
	$(TBLGEN) --gen-enum-defs $(INCLUDES) $< -o $@

TupleOps.hpp.inc: TupleOps.td
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

TupleOps.cpp.inc: TupleOps.td
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

TupleTypes.hpp.inc: TupleTypes.td
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

TupleTypes.cpp.inc: TupleTypes.td
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

libtuple_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

PLUGIN_OBJECTS := $(DIALECT_OBJECTS) Plugin.o

libtuple_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

.PHONY: test
test: libtuple_dialect.so
	@echo "Running tuple dialect tests..."
	lit tests

clean:
	rm -f *.o *.inc *.a *.so
