# NeoGPU build (native aarch64)
CC = gcc
CFLAGS = -O3 -march=armv8.2-a+fp16+simd -mtune=cortex-a72 \
         -ffast-math -funroll-loops -Wall -Wextra -DNDEBUG -Iinclude
LDFLAGS = -lm -lGLESv2 -lgbm -ldrm -lEGL
STRIP = strip

SRC_DIR = src
OBJ_DIR = src

HS_SRC = $(SRC_DIR)/hs_core.c $(SRC_DIR)/hs_nodes.c $(SRC_DIR)/hs_gpu.c $(SRC_DIR)/main.c
HS_OBJ = $(HS_SRC:.c=.o)
HS_TARGET = neogpu_demo

.PHONY: build clean run all test

all: build

build: $(HS_TARGET)

$(HS_TARGET): $(HS_OBJ)
	$(CC) $(HS_OBJ) $(LDFLAGS) -o $@
	$(STRIP) $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(HS_OBJ) $(HS_TARGET)

run: $(HS_TARGET)
	./$(HS_TARGET)

test: $(HS_OBJ)
	@if [ -z "$(N)" ]; then \
		echo "Usage: make test N=XX"; \
		echo "  where XX = 01, 02, 03, 04, 05, or 06"; \
		echo "Example: make test N=06"; \
		exit 1; \
	fi
	$(CC) $(CFLAGS) -c tests/test_$(N)_*.c -o /tmp/test_$(N).o
	$(CC) $(SRC_DIR)/hs_core.o $(SRC_DIR)/hs_nodes.o $(SRC_DIR)/hs_gpu.o /tmp/test_$(N).o $(LDFLAGS) -o /tmp/test_$(N)
	sudo /tmp/test_$(N)
