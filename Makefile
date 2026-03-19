# NeoGPU build (native aarch64)
CC = gcc
CFLAGS = -O3 -march=armv8-a+simd -mtune=cortex-a72 \
         -ffast-math -funroll-loops -Wall -Wextra -DNDEBUG -Iinclude \
         -MMD -MP
LDFLAGS = -lm -lGLESv2 -lgbm -ldrm -lEGL -lpthread
STRIP = strip

VERSION := $(shell cat VERSION 2>/dev/null || echo "0.0.0")

SRC_DIR = src
OBJ_DIR = src

HS_SRC = $(SRC_DIR)/hs_core.c $(SRC_DIR)/hs_nodes.c $(SRC_DIR)/hs_gpu.c $(SRC_DIR)/hs_async.c $(SRC_DIR)/hs_backend_gles.c $(SRC_DIR)/hs_ipc.c $(SRC_DIR)/hs_ml.c $(SRC_DIR)/hs_ml_loader.c $(SRC_DIR)/main.c
HS_OBJ = $(HS_SRC:.c=.o)
HS_DEPS = $(HS_OBJ:.o=.d)
HS_TARGET = neogpu_demo

TESTS = test_01_clear test_02_triangle test_03_instancing test_04_blending test_05_cube3d test_06_raycast test_07_message_triangle

.PHONY: build clean run all test run-test build-tests version

all: build build-tests

version:
	@echo "NeoGPU v$(VERSION)"

build: $(HS_TARGET)

-include $(HS_DEPS)

$(HS_TARGET): $(HS_OBJ)
	$(CC) $(HS_OBJ) $(LDFLAGS) -o $@
	$(STRIP) $@

TOOL_TGT = neogpu_tool

neogpu_pong: tools/neogpu_pong.c
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o -o $@ $(LDFLAGS)
	$(STRIP) $@

neogpu_capture_demo: tools/neogpu_capture_demo.c
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_backend_gles.o -o $@ $(LDFLAGS)
	$(STRIP) $@

neogpu_pong_ai: tools/neogpu_pong_ai.c
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_backend_gles.o -o $@ $(LDFLAGS)
	$(STRIP) $@

neogpu_pong_llm: tools/neogpu_pong_llm.c src/hs_ml_ternary_neon.o src/hs_ml_ternary_coproc.o src/hs_ml_ternary_mt.o src/hs_ml_gpu_gemm.o
	$(CC) $(CFLAGS) $< src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_backend_gles.o src/hs_ml_infer.o src/hs_ml_loader.o src/hs_ml_loader_ternary.o src/hs_ml.o src/hs_ml_ternary.o src/hs_ml_msg.o src/hs_ml_routing.o src/hs_ml_binary.o src/hs_ml_ternary_neon.o src/hs_ml_ternary_coproc.o src/hs_ml_ternary_mt.o src/hs_ml_gpu_gemm.o -o $@ $(LDFLAGS)
	$(STRIP) $@

.PHONY: tool

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(HS_OBJ) $(HS_DEPS) $(HS_TARGET) tests/*.o tests/test_01_clear tests/test_02_triangle tests/test_03_instancing tests/test_04_blending tests/test_05_cube3d tests/test_06_raycast tests/test_07_message_triangle neogpu_tool neogpu_pong

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
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_06_raycast.o $(LDFLAGS) -o tests/test_06_raycast
	$(STRIP) tests/test_06_raycast
	$(CC) $(CFLAGS) -c tests/test_07_message_triangle.c -o tests/test_07_message_triangle.o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o $(SRC_DIR)/hs_backend_gles.o tests/test_07_message_triangle.o $(LDFLAGS) -o tests/test_07_message_triangle
	$(STRIP) tests/test_07_message_triangle

run-test:
	@if [ -z "$(N)" ]; then \
		echo "Usage: make run-test N=XX"; \
		echo "  where XX = 01, 02, 03, 04, 05, or 06"; \
		echo "Example: make run-test N=06"; \
		exit 1; \
	fi
	$(CC) $(CFLAGS) -c tests/test_$(N)_*.c -o /tmp/test_$(N).o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o $(SRC_DIR)/hs_async.o /tmp/test_$(N).o $(LDFLAGS) -o /tmp/test_$(N)
	sudo /tmp/test_$(N)

release:
	@echo "Creating release v$(VERSION)"
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@echo "Run 'git push --tags' to push tags"
