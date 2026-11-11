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

# --- Core: graphics + runtime substrate ONLY (no ML). This is the project's
#     focus and builds with zero inference dependencies. ---
HS_SRC = $(SRC_DIR)/hs_core.c $(SRC_DIR)/hs_nodes.c $(SRC_DIR)/hs_gpu.c $(SRC_DIR)/hs_async.c $(SRC_DIR)/hs_backend_gles.c $(SRC_DIR)/hs_ipc.c $(SRC_DIR)/hs_text.c $(SRC_DIR)/main.c

# --- Optional experimental ML/inference (BitNet). Opt in with `make ml`.
#     NOT part of the default build. See docs/ML_API.md.
#
#     The ML build compiles into its own object tree and produces its own
#     binary. This is deliberate: hs_ipc.c is compiled differently when
#     HS_ENABLE_ML is defined, so sharing objects between the two builds
#     would let a plain `make` silently reuse ML-enabled objects (or the
#     reverse), quietly breaking the core/ML separation.
ML_SRC = $(SRC_DIR)/hs_ml.c $(SRC_DIR)/hs_ml_loader.c $(SRC_DIR)/hs_ml_ternary_mt.c

# --- The real ML product: the BitNet chat tool. It links the whole
#     ternary inference stack -- the GGUF loader, the LUT projection
#     kernels, the two-stage lm_head, the I2_S layout handling and the
#     GPU GEMM. The `ml` target above does NOT: it builds the older
#     routing stack, which the chat tool does not use. Without this
#     target, a compile error in the ternary stack reaches no build at
#     all, CI included.
ML_CHAT_SRC = $(SRC_DIR)/hs_ml_loader.c $(SRC_DIR)/hs_ml_loader_ternary.c \
              $(SRC_DIR)/hs_ml_infer.c $(SRC_DIR)/hs_ml_ternary_neon.c \
              $(SRC_DIR)/hs_ml_i2s_layout.c $(SRC_DIR)/hs_ml.c \
              $(SRC_DIR)/hs_ml_gpu_gemm.c $(SRC_DIR)/hs_ml_msg.c \
              $(SRC_DIR)/hs_ml_routing.c $(SRC_DIR)/hs_ml_binary.c \
              $(SRC_DIR)/hs_ml_ternary_mt.c $(SRC_DIR)/hs_ml_ternary.c
ML_CHAT_TARGET = neogpu_bitnet_chat
HS_SRC_NO_MAIN = $(filter-out $(SRC_DIR)/main.c,$(HS_SRC))

ML_BUILD_DIR = build/ml
ML_CFLAGS = $(CFLAGS) -DHS_ENABLE_ML
ML_FULL_SRC = $(HS_SRC) $(ML_SRC)
ML_FULL_OBJ = $(patsubst $(SRC_DIR)/%.c,$(ML_BUILD_DIR)/%.o,$(ML_FULL_SRC))
ML_FULL_DEPS = $(ML_FULL_OBJ:.o=.d)
ML_TARGET = neogpu_demo_ml

HS_OBJ = $(HS_SRC:.c=.o)
HS_DEPS = $(HS_OBJ:.o=.d)
HS_TARGET = neogpu_demo

TESTS = test_01_clear test_02_triangle test_03_instancing test_04_blending test_05_cube3d test_06_raycast test_07_message_triangle test_08_raycast_orbit

.PHONY: build clean run all test run-test build-tests version ml ml-chat test-i8-bounds test-ternary-formats test-code-mapping test-i2s-repack test-i2s-upstream

all: build build-tests

version:
	@echo "NeoGPU v$(VERSION)"

build: $(HS_TARGET)

# Opt-in experimental ML/inference build: the core sources plus the ML
# sources, compiled with HS_ENABLE_ML so hs_ipc.c builds in its ML tool-bus
# ops. Uses a separate object tree and a separate binary (neogpu_demo_ml) so
# it can never contaminate -- or be contaminated by -- the core build.
ml: $(ML_TARGET)

-include $(HS_DEPS)
-include $(ML_FULL_DEPS)

$(HS_TARGET): $(HS_OBJ)
	$(CC) $(HS_OBJ) $(LDFLAGS) -o $@
	$(STRIP) $@

$(ML_BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(ML_BUILD_DIR)
	$(CC) $(ML_CFLAGS) -c $< -o $@

$(ML_TARGET): $(ML_FULL_OBJ)
	$(CC) $(ML_FULL_OBJ) $(LDFLAGS) -o $@
	$(STRIP) $@

ml-chat: $(ML_CHAT_TARGET)

ML_CHAT_OBJ = $(patsubst $(SRC_DIR)/%.c,$(ML_BUILD_DIR)/%.o,$(HS_SRC_NO_MAIN) $(ML_CHAT_SRC))

$(ML_BUILD_DIR)/neogpu_bitnet_chat.o: tools/neogpu_bitnet_chat.c
	@mkdir -p $(ML_BUILD_DIR)
	$(CC) $(ML_CFLAGS) -c $< -o $@

$(ML_CHAT_TARGET): $(ML_BUILD_DIR)/neogpu_bitnet_chat.o $(ML_CHAT_OBJ)
	$(CC) $(ML_BUILD_DIR)/neogpu_bitnet_chat.o $(ML_CHAT_OBJ) $(LDFLAGS) -o $@
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

# Pong + real BitNet inference. Experimental: this links the ML objects, which
# are NOT part of the core build. Every object in the link line is declared as
# a prerequisite so the target builds from a clean tree.
PONG_LLM_OBJ = src/hs_core.o src/hs_nodes.o src/hs_gpu.o src/hs_backend_gles.o \
               src/hs_ml_infer.o src/hs_ml_loader.o src/hs_ml_loader_ternary.o \
               src/hs_ml.o src/hs_ml_ternary.o src/hs_ml_msg.o src/hs_ml_routing.o \
               src/hs_ml_binary.o src/hs_ml_ternary_neon.o src/hs_ml_ternary_coproc.o \
               src/hs_ml_ternary_mt.o src/hs_ml_gpu_gemm.o

neogpu_pong_llm: tools/neogpu_pong_llm.c $(PONG_LLM_OBJ)
	$(CC) $(CFLAGS) $< $(PONG_LLM_OBJ) -o $@ $(LDFLAGS)
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
	rm -f $(HS_OBJ) $(HS_DEPS) $(HS_TARGET) tests/*.o tests/test_01_clear tests/test_02_triangle tests/test_03_instancing tests/test_04_blending tests/test_05_cube3d tests/test_06_raycast tests/test_07_message_triangle tests/test_08_raycast_orbit neogpu_tool neogpu_pong neogpu_pong_ai neogpu_pong_llm neogpu_capture_demo neogpu_procgame
	rm -f tests/test_01 tests/test_02 tests/test_03 tests/test_04 tests/test_05 tests/test_06 tests/test_07 tests/test_08
	rm -f tests/redteam_lmhead_i8_bounds tests/hs_ml_ternary_neon_asan.o
	rm -f tests/redteam_ternary_formats tests/hs_ml_ternary_neon_formats.o
	rm -f tests/redteam_code_mapping
	rm -f tests/redteam_i2s_repack tests/redteam_i2s_upstream
	rm -rf $(ML_BUILD_DIR) $(ML_TARGET) $(ML_CHAT_TARGET)

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

# Memory-safety regression for the int8 lm_head compatibility entry point.
# Built with AddressSanitizer so an out-of-bounds access is reported instead
# of silently corrupting memory. Needs a sanitizer-capable compiler.
test-i8-bounds:
	$(CC) $(CFLAGS) -fsanitize=address -g -c $(SRC_DIR)/hs_ml_ternary_neon.c -o tests/hs_ml_ternary_neon_asan.o
	$(CC) $(CFLAGS) -fsanitize=address -g tests/redteam_lmhead_i8_bounds.c tests/hs_ml_ternary_neon_asan.o $(LDFLAGS) -o tests/redteam_lmhead_i8_bounds
	./tests/redteam_lmhead_i8_bounds

# Layout and format checks for the ternary weight formats. Pins the claims
# made in docs/interface/TERNARY_WEIGHT_FORMATS.md: the production LUT
# kernel's decode, the three bit layouts, the 8-plane base-3 expansion,
# and the source lines the documentation depends on.
# Links only libm and libpthread -- it is a CPU-only test and does not
# need the graphics libraries.
test-ternary-formats:
	$(CC) $(CFLAGS) -c $(SRC_DIR)/hs_ml_ternary_neon.c -o tests/hs_ml_ternary_neon_formats.o
	$(CC) $(CFLAGS) tests/redteam_ternary_formats.c tests/hs_ml_ternary_neon_formats.o -lm -lpthread -o tests/redteam_ternary_formats
	./tests/redteam_ternary_formats

# The ternary code-mapping boundary. There are two 2-bit code mappings in
# this repository, and each packer must be read by the decoder that uses
# the matching one. CPU-only: links nothing but libm.
test-code-mapping:
	$(CC) $(CFLAGS) tests/redteam_code_mapping.c -lm -o tests/redteam_code_mapping
	./tests/redteam_code_mapping

# The I2_S byte-layout conversion. Upstream BitNet packs I2_S as
# 64-weight blocks; NeoGPU's kernels pack four consecutive weights per
# byte. The loader converts once, at load time; this pins that the
# converted bytes are byte-identical to natively-packed ones.
test-i2s-repack:
	$(CC) $(CFLAGS) $(SRC_DIR)/hs_ml_i2s_layout.c tests/redteam_i2s_repack.c -lm -o tests/redteam_i2s_repack
	./tests/redteam_i2s_repack

# Upstream BitNet's own I2_S producer and consumer, extracted from the
# vendored source at build time, so the conversion is checked against
# the reference implementation and not against a copy of it that could
# drift. See tools/extract_upstream_i2s.py.
UPSTREAM_I2S_SRC = experimental/BitNet/src/ggml-bitnet-mad.cpp
UPSTREAM_I2S_REF = $(ML_BUILD_DIR)/upstream_i2s_ref.c

$(UPSTREAM_I2S_REF): $(UPSTREAM_I2S_SRC) tools/extract_upstream_i2s.py
	@mkdir -p $(ML_BUILD_DIR)
	python3 tools/extract_upstream_i2s.py $(UPSTREAM_I2S_SRC) $@

test-i2s-upstream: $(UPSTREAM_I2S_REF)
	$(CC) $(CFLAGS) -Wno-unused-parameter -I$(ML_BUILD_DIR) $(SRC_DIR)/hs_ml_i2s_layout.c tests/redteam_i2s_upstream.c -lm -o tests/redteam_i2s_upstream
	./tests/redteam_i2s_upstream

run-test:
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
