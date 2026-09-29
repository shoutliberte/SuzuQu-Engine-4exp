CC ?= gcc
HIPCC ?= /opt/rocm/bin/hipcc
ROCM_ARCH ?= gfx1100
# /etc/profile.d/rocm.sh sets this in login shells; non-login shells need it.
export ROCM_PATH ?= /opt/rocm
CFLAGS ?= -O2 -g -Wall -Wextra -Wshadow -Wno-unused-parameter -pthread -fopenmp
CFLAGS += -I src
HIPFLAGS ?= -O3 -ffast-math -fno-finite-math-only -std=c++17 --offload-arch=$(ROCM_ARCH)
HIPFLAGS += -D__HIP_PLATFORM_AMD__ -DQ4_HIP -I src -Wno-unused-parameter
LDFLAGS ?= -pthread -fopenmp -lm

SRC = src/q4_gguf.c src/q4_place.c src/q4_expert.c src/q4_ple.c src/q4_quant.c src/q4_moe.c src/q4_store.c src/q4_fwd.c src/q4_tok.c src/q4_engine.c src/q4_mtp.c src/q4_server.c src/q4_crash.c
OBJ = $(SRC:.c=.o)
HIP_OBJ = src/q4_rocm.o
CPUEXP_OBJ = src/q4_cpuexp.o

.PHONY: all test clean inspect plan

all: q4 q4-test

q4: $(OBJ) src/q4_cli.o $(HIP_OBJ) $(CPUEXP_OBJ)
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(LDFLAGS)

q4-test: $(OBJ) src/q4_test.o $(HIP_OBJ) $(CPUEXP_OBJ)
	$(HIPCC) $(HIPFLAGS) -o $@ $^ $(LDFLAGS)

src/q4_cpuexp.o: src/q4_cpuexp.c src/q4.h
	$(CC) $(CFLAGS) -O3 -mavx2 -mfma -mf16c -c -o $@ $<

src/%.o: src/%.c src/q4.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/q4_rocm.o: src/q4_rocm.cu src/q4.h
	$(HIPCC) $(HIPFLAGS) -c -o $@ $<

MODEL_NAME := IQ3E-Q8D-MTP/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf
MODEL ?= $(firstword $(wildcard models/$(MODEL_NAME) ../models/$(MODEL_NAME)) models/$(MODEL_NAME))

test: q4-test
	./q4-test
	./q4-test "$(MODEL)"

inspect: q4
	./q4 inspect "$(MODEL)"

plan: q4
	./q4 plan "$(MODEL)" --ctx 8192

clean:
	rm -f q4 q4-test src/*.o
