# NeoGPU build (native aarch64)
CC = gcc
CFLAGS = -O3 -march=armv8-a+simd -mtune=cortex-a72 \
         -ffast-math -funroll-loops -Wall -Wextra -DNDEBUG -Iinclude \
         -MMD -MP

# libdrm installs its headers under /usr/include/libdrm on Debian/Ubuntu,
# and xf86drm.h does #include <drm.h> -- so that directory has to be on the
# include path or the build fails at src/hs_async.c. gcc ignores a -I path
# that does not exist, so this is harmless on systems that lay the headers
# out differently (and on the Pi).
CFLAGS += -I/usr/include/libdrm

LDFLAGS = -lm -lGLESv2 -lgbm -ldrm -lEGL -lpthread
STRIP = strip

VERSION := $(shell cat VERSION 2>/dev/null || echo "0.0.0")

SRC_DIR = src
OBJ_DIR = src

# --- Graphics and runtime substrate ---
HS_SRC = $(SRC_DIR)/hs_core.c $(SRC_DIR)/hs_nodes.c $(SRC_DIR)/hs_gpu.c $(SRC_DIR)/hs_async.c $(SRC_DIR)/hs_backend_gles.c $(SRC_DIR)/hs_ipc.c $(SRC_DIR)/hs_text.c $(SRC_DIR)/hs_input_stream.c $(SRC_DIR)/main.c

HS_OBJ = $(HS_SRC:.c=.o)
HS_DEPS = $(HS_OBJ:.o=.d)
HS_TARGET = neogpu_demo

TESTS = test_01_clear test_02_triangle test_03_instancing test_04_blending test_05_cube3d test_06_raycast test_07_message_triangle test_08_raycast_orbit

.PHONY: build clean run all test run-test build-tests version sanitize tsan check-sources

all: build build-tests

version:
	@echo "NeoGPU v$(VERSION)"

build: $(HS_TARGET)

test: $(HS_TARGET) tests/test_p0 neogpu_tool
	./$(HS_TARGET) --check
	./tests/test_p0
	python3 tests/test_cli.py

.PHONY: benchmark
benchmark: $(HS_TARGET)
	./$(HS_TARGET) --benchmark

tests/test_p0: tests/test_p0.c $(filter-out src/main.o,$(HS_OBJ))
	$(CC) $(CFLAGS) tests/test_p0.c $(filter-out src/main.o,$(HS_OBJ)) $(LDFLAGS) -o $@

SAN_FLAGS = -O1 -g -fno-omit-frame-pointer -Iinclude -I/usr/include/libdrm
LIB_SRC = $(filter-out src/main.c,$(HS_SRC))

sanitize: $(HS_TARGET)
	$(CC) $(SAN_FLAGS) -fsanitize=address,undefined tests/test_p0.c $(LIB_SRC) $(LDFLAGS) -o tests/test_p0_sanitize
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ./tests/test_p0_sanitize
	$(CC) $(SAN_FLAGS) -fsanitize=address,undefined src/main.c $(LIB_SRC) $(LDFLAGS) -o tests/test_headless_sanitize
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ./tests/test_headless_sanitize --check
	$(CC) $(SAN_FLAGS) -fsanitize=address,undefined tools/neogpu_tool.c $(LDFLAGS) -o tests/test_tool_sanitize
	NEOGPU_TOOL=./tests/test_tool_sanitize ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 python3 tests/test_cli.py

tsan: $(HS_TARGET)
	$(CC) $(SAN_FLAGS) -fsanitize=thread tests/test_p0.c $(LIB_SRC) $(LDFLAGS) -o tests/test_p0_tsan
	TSAN_OPTIONS=halt_on_error=1 ./tests/test_p0_tsan
	$(CC) $(SAN_FLAGS) -fsanitize=thread src/main.c $(LIB_SRC) $(LDFLAGS) -o tests/test_headless_tsan
	TSAN_OPTIONS=halt_on_error=1 ./tests/test_headless_tsan
	$(CC) $(SAN_FLAGS) -fsanitize=thread tools/neogpu_tool.c $(LDFLAGS) -o tests/test_tool_tsan
	NEOGPU_TOOL=./tests/test_tool_tsan TSAN_OPTIONS=halt_on_error=1 python3 tests/test_cli.py

check-sources:
	@set -e; for source in $(wildcard src/*.c tests/*.c tools/*.c); do \
		echo "Checking $$source"; $(CC) $(CFLAGS) -Werror -fsyntax-only $$source; \
	done

.PHONY: test-hardware
test-hardware: build-tests tests/test_evdev
	NEOGPU_CAPTURE=tests/p0-clear.ppm ./tests/test_01_clear
	NEOGPU_CAPTURE=tests/p0-direct-triangle.ppm ./tests/test_02_triangle
	NEOGPU_CAPTURE=tests/p0-instancing.ppm ./tests/test_03_instancing
	NEOGPU_CAPTURE=tests/p0-blending.ppm ./tests/test_04_blending
	NEOGPU_CAPTURE=tests/p0-cube.ppm ./tests/test_05_cube3d
	NEOGPU_CAPTURE=tests/p0-raycast.ppm ./tests/test_06_raycast
	NEOGPU_CAPTURE=tests/p0-triangle.ppm ./tests/test_07_message_triangle
	NEOGPU_CAPTURE=tests/p0-orbit.ppm ./tests/test_08_raycast_orbit
	./tests/test_evdev

tests/test_evdev: tests/test_evdev.c src/hs_input_stream.o
	$(CC) $(CFLAGS) tests/test_evdev.c src/hs_input_stream.o $(LDFLAGS) -o $@

-include $(HS_DEPS)
-include $(wildcard tests/*.d tools/*.d *.d)

$(HS_TARGET): $(HS_OBJ)
	$(CC) $(HS_OBJ) $(LDFLAGS) -o $@
	$(STRIP) $@

TOOL_TGT = neogpu_tool

neogpu_pong: tools/neogpu_pong.c src/hs_core.o src/hs_nodes.o src/hs_text.o
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o src/hs_text.o -o $@ $(LDFLAGS)
	$(STRIP) $@

neogpu_capture_demo: tools/neogpu_capture_demo.c src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_backend_gles.o
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_backend_gles.o -o $@ $(LDFLAGS)
	$(STRIP) $@

neogpu_pong_ai: tools/neogpu_pong_ai.c src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_backend_gles.o
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_backend_gles.o -o $@ $(LDFLAGS)
	$(STRIP) $@

neogpu_procgame: tools/neogpu_procgame.c src/hs_core.o src/hs_nodes.o
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o -o $@ $(LDFLAGS)
	$(STRIP) $@

.PHONY: tool

# Standalone IPC client. Links only the core message layer; the request/response
# packing helpers it uses are static inline in include/hs_msg.h, so no other
# objects are required.
neogpu_tool: tools/neogpu_tool.c src/hs_core.o src/hs_nodes.o src/hs_text.o
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o src/hs_text.o -o $@ $(LDFLAGS)
	$(STRIP) $@

tool: neogpu_tool

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f tests/test_p0 tests/test_p0_sanitize tests/test_p0_tsan tests/test_headless_sanitize tests/test_headless_tsan tests/*.d tools/*.d *.d
	rm -f tests/test_evdev
	rm -f tests/test_tool_sanitize tests/test_tool_tsan
	rm -f $(HS_OBJ) $(HS_DEPS) $(HS_TARGET) tests/*.o tests/test_01_clear tests/test_02_triangle tests/test_03_instancing tests/test_04_blending tests/test_05_cube3d tests/test_06_raycast tests/test_07_message_triangle tests/test_08_raycast_orbit neogpu_tool neogpu_pong neogpu_pong_ai neogpu_capture_demo neogpu_procgame
	rm -f tests/test_01 tests/test_02 tests/test_03 tests/test_04 tests/test_05 tests/test_06 tests/test_07 tests/test_08

run: $(HS_TARGET)
	./$(HS_TARGET)

build-tests: $(HS_OBJ)
	@echo "Building graphics tests..."
	$(CC) $(CFLAGS) -c tests/test_01_clear.c -o tests/test_01_clear.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_01_clear.o $(LDFLAGS) -o tests/test_01_clear
	$(STRIP) tests/test_01_clear
	$(CC) $(CFLAGS) -c tests/test_02_triangle.c -o tests/test_02_triangle.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_02_triangle.o $(LDFLAGS) -o tests/test_02_triangle
	$(STRIP) tests/test_02_triangle
	$(CC) $(CFLAGS) -c tests/test_03_instancing.c -o tests/test_03_instancing.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_03_instancing.o $(LDFLAGS) -o tests/test_03_instancing
	$(STRIP) tests/test_03_instancing
	$(CC) $(CFLAGS) -c tests/test_04_blending.c -o tests/test_04_blending.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_04_blending.o $(LDFLAGS) -o tests/test_04_blending
	$(STRIP) tests/test_04_blending
	$(CC) $(CFLAGS) -c tests/test_05_cube3d.c -o tests/test_05_cube3d.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_05_cube3d.o $(LDFLAGS) -o tests/test_05_cube3d
	$(STRIP) tests/test_05_cube3d
	$(CC) $(CFLAGS) -c tests/test_06_raycast.c -o tests/test_06_raycast.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o $(SRC_DIR)/hs_text.o tests/test_06_raycast.o $(LDFLAGS) -o tests/test_06_raycast
	$(STRIP) tests/test_06_raycast
	$(CC) $(CFLAGS) -c tests/test_07_message_triangle.c -o tests/test_07_message_triangle.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_07_message_triangle.o $(LDFLAGS) -o tests/test_07_message_triangle
	$(STRIP) tests/test_07_message_triangle
	$(CC) $(CFLAGS) -c tests/test_08_raycast_orbit.c -o tests/test_08_raycast_orbit.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_08_raycast_orbit.o $(LDFLAGS) -o tests/test_08_raycast_orbit
	$(STRIP) tests/test_08_raycast_orbit
	$(STRIP) tests/test_07_message_triangle

run-test: $(HS_OBJ)
	@if [ -z "$(N)" ]; then \
		echo "Usage: make run-test N=XX"; \
		echo "  where XX = 01, 02, 03, 04, 05, 06, 07, or 08"; \
		echo "Example: make run-test N=06"; \
		exit 1; \
	fi
	$(CC) $(CFLAGS) -c tests/test_$(N)_*.c -o tests/test_$(N).o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o $(SRC_DIR)/hs_text.o tests/test_$(N).o $(LDFLAGS) -o tests/test_$(N)
	$(STRIP) tests/test_$(N)
	sudo tests/test_$(N)

release:
	@echo "Creating release v$(VERSION)"
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@echo "Run 'git push --tags' to push tags"
