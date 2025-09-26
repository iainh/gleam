ROOT_DIR     := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
SRC_DIR      ?= $(abspath $(ROOT_DIR)/..)
FACIL_DIR    ?= $(ROOT_DIR)/facil.io
FACIL_DIR    := $(abspath $(FACIL_DIR))
BUILD_DIR    := $(ROOT_DIR)/build
C_SOURCES    := $(SRC_DIR)/src/facil_io_ffi.c
C_OBJECTS    := $(BUILD_DIR)/facil_io_ffi.o
STATIC_LIB   := $(BUILD_DIR)/libfacil_io.a
RUNTIME_INC  := $(abspath $(ROOT_DIR)/../../../runtime-cranelift/include)

LIB_INCLUDE_DIRS := $(shell find $(FACIL_DIR)/lib -type d 2>/dev/null)
LIB_INCLUDE_FLAGS := $(foreach dir,$(LIB_INCLUDE_DIRS),-I$(dir))

CFLAGS       := -std=c11 -O2 -fPIC -Wall -Wextra -Wno-unused-parameter -I$(RUNTIME_INC)
CFLAGS       += -I$(FACIL_DIR)/include -I$(FACIL_DIR)/src -I$(FACIL_DIR)/lib $(LIB_INCLUDE_FLAGS)

FACIL_LIBS   := $(FACIL_DIR)/build/libfacil.a \
                $(FACIL_DIR)/build/libfiobj.a \
                $(FACIL_DIR)/build/libfio.a

.PHONY: all clean facil libs objects archive clone

all: $(STATIC_LIB)

clone:
	@mkdir -p $(dir $(FACIL_DIR))
	@if [ ! -d "$(FACIL_DIR)" ]; then \
		echo "Cloning facil.io into $(FACIL_DIR)"; \
		git clone git@github.com:boazsegev/facil.io.git "$(FACIL_DIR)"; \
	fi

facil: clone
	@echo "Building facil.io library objects"
	@$(MAKE) -C "$(FACIL_DIR)" lib

libs: facil
	@mkdir -p $(BUILD_DIR)

objects: libs $(C_OBJECTS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/src/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c "$<" -o "$@"

$(STATIC_LIB): objects
	@echo "Creating archive $(STATIC_LIB)"
	@rm -f "$(STATIC_LIB)"
	@objects=$$(find "$(FACIL_DIR)/tmp/lib" -name '*.o' -print); \
	libtool -static -o "$(STATIC_LIB)" $(C_OBJECTS) $$objects

clean:
	@rm -rf "$(BUILD_DIR)"
