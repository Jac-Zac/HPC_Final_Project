# Force mpicc and mpich to use gcc-15
export OMPI_CC := gcc
export MPICH_CC := gcc

# Compiler
CC = gcc
MPICC = mpicc

# Compilation flags
# CFLAGS = -O1 -Wall -Wextra -fopenmp -Iinclude -fno-tree-vectorize -g
CFLAGS = -O3 -Wall -Wextra -march=native -fopenmp -Iinclude -g

# Source files
PARALLEL_SRC = src/stencil_parallel.c
SERIAL_SRC   = src/stencil_serial.c

# Default mode: parallel
MODE ?= parallel

# Select source and output basename based on mode
ifeq ($(MODE),parallel)
	SRC = $(PARALLEL_SRC)
	BASENAME = stencil_parallel
else ifeq ($(MODE),serial)
	SRC = $(SERIAL_SRC)
	BASENAME = stencil_serial
else
	$(error Invalid MODE specified. Use MODE=parallel or MODE=serial)
endif

# Default target
all: $(BASENAME)

# Compile rule
$(BASENAME):
	$(MPICC) $(CFLAGS) $(SRC) -o $(BASENAME)

test: all
	mpirun -np 4 stencil_parallel -x 100 -y 100 -n 50 -o 1 -e 4 -s 1337                                                                                                  1m 4s 
	pytest -v tests
	rm -f stencil_parallel stencil_serial *.bin

# Clean built executables
clean:
	rm -f stencil_parallel stencil_serial *.bin

# Help target
help:
	@echo "Usage:"
	@echo "  make [target] [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  all            Build the default mode (parallel or serial)"
	@echo "  stencil_parallel  Build the parallel version"
	@echo "  stencil_serial    Build the serial version"
	@echo "  test           Run tests with pytest"
	@echo "  clean          Remove built executables and .bin files"
	@echo "  help           Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  MODE=parallel|serial    Select which version to build (default: parallel)"

.PHONY: all clean help test
