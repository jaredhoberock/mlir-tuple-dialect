# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

ifndef MLIR_SYS_220_PREFIX
$(error MLIR_SYS_220_PREFIX must be set; point it at your LLVM/MLIR 22 install)
endif

BUILD_DIR := build
TRAIT_DIALECT_SOURCE_DIR := ../../mlir-trait-dialect/cpp
TRAIT_DIALECT_INCLUDE_DIR := ../../mlir-trait-dialect/cpp/build
LLVM_BIN_PATH := $(MLIR_SYS_220_PREFIX)/bin
LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen

CXX := clang++
CXXFLAGS := -Wall -Werror -fPIC $(shell $(LLVM_CONFIG) --cxxflags)
INCLUDES := -I$(BUILD_DIR) -I. -I$(MLIR_SYS_220_PREFIX)/include -I$(TRAIT_DIALECT_SOURCE_DIR) -I$(TRAIT_DIALECT_INCLUDE_DIR)

DIALECT_SOURCES := c_api.cpp Canonicalization.cpp ConvertToLLVM.cpp Elaboration.cpp ImplGenerators.cpp Monomorphization.cpp Tuple.cpp TupleEnums.cpp TupleOps.cpp TupleTypes.cpp
DIALECT_OBJECTS := $(addprefix $(BUILD_DIR)/,$(DIALECT_SOURCES:.cpp=.o))
PLUGIN_OBJECTS := $(DIALECT_OBJECTS) $(BUILD_DIR)/Plugin.o
TD_FILES := $(wildcard *.td)

GENERATED := \
  $(BUILD_DIR)/Tuple.hpp.inc \
  $(BUILD_DIR)/Tuple.cpp.inc \
  $(BUILD_DIR)/TupleEnums.hpp.inc \
  $(BUILD_DIR)/TupleEnums.cpp.inc \
  $(BUILD_DIR)/TupleOps.hpp.inc \
  $(BUILD_DIR)/TupleOps.cpp.inc \
  $(BUILD_DIR)/TupleTypes.hpp.inc \
  $(BUILD_DIR)/TupleTypes.cpp.inc

.PHONY: all clean test

all: $(BUILD_DIR)/libtuple_dialect.a $(BUILD_DIR)/libtuple_dialect.so

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/Tuple.hpp.inc: Tuple.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/Tuple.cpp.inc: Tuple.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/TupleEnums.hpp.inc: TupleEnums.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-enum-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/TupleEnums.cpp.inc: TupleEnums.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-enum-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/TupleOps.hpp.inc: TupleOps.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/TupleOps.cpp.inc: TupleOps.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/TupleTypes.hpp.inc: TupleTypes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/TupleTypes.cpp.inc: TupleTypes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/%.o: %.cpp $(GENERATED) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/libtuple_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

$(BUILD_DIR)/libtuple_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

test: $(BUILD_DIR)/libtuple_dialect.so
	TUPLE_DIALECT_PLUGIN=$(CURDIR)/$(BUILD_DIR)/libtuple_dialect.so lit tests

clean:
	rm -rf $(BUILD_DIR)
