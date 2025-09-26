ROOT_DIR     := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
FACIL_DIR    ?= $(ROOT_DIR)/vendor/facil.io
FACIL_DIR    := $(abspath $(FACIL_DIR))
BUILD_DIR    := $(ROOT_DIR)/build
WRAPPER_SRC  := $(ROOT_DIR)/facil_wrapper.c
WRAPPER_OBJ  := $(BUILD_DIR)/facil_wrapper.o
STATIC_LIB   := $(BUILD_DIR)/libfacil_io_demo.a

CFLAGS       := -std=c11 -O2 -fPIC -Wall -Wextra -Wno-unused-parameter
CFLAGS       += -I$(FACIL_DIR)/include -I$(FACIL_DIR)/src -I$(FACIL_DIR)/lib

FACIL_LIBS   := $(FACIL_DIR)/build/libfacil.a \
                $(FACIL_DIR)/build/libfiobj.a \
                $(FACIL_DIR)/build/libfio.a

.PHONY: all clean facil libs wrapper archive clone

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

wrapper: libs
	@INCLUDES=""; \
	for dir in $$(find "$(FACIL_DIR)/lib" -type d -print); do \
		INCLUDES="$$INCLUDES -I$$dir"; \
	done; \
	$(CC) $(CFLAGS) $$INCLUDES -c "$(WRAPPER_SRC)" -o "$(WRAPPER_OBJ)"

$(STATIC_LIB): wrapper
	@echo "Creating archive $(STATIC_LIB)"
	@rm -f "$(STATIC_LIB)"
	@objects=$$(find "$(FACIL_DIR)/tmp/lib" -name '*.o' -print); \
	libtool -static -o "$(STATIC_LIB)" "$(WRAPPER_OBJ)" $$objects

clean:
	@rm -rf "$(BUILD_DIR)"
