# NeoGPU build (native aarch64)
CC = gcc
CFLAGS = -O3 -march=armv8.2-a+fp16+simd -mtune=cortex-a72 \
         -ffast-math -funroll-loops -Wall -Wextra -DNDEBUG
LDFLAGS = -lm
STRIP = strip

HS_SRC = hs_core.c hs_nodes.c hs_gpu.c main.c
HS_OBJ = $(HS_SRC:.c=.o)
HS_TARGET = neogpu_demo

.PHONY: build clean run all

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
