# Force mpicc and mpich to use gcc-15
export OMPI_CC := gcc
export MPICH_CC := gcc

# Compiler
CC = gcc
MPICC = mpicc

# Compilation flags
# CFLAGS = -O1 -Wall -Wextra -fopenmp -Iinclude -fno-tree-vectorize
CFLAGS = -O3 -Wall -Wextra -march=native -fopenmp -Iinclude

# Enable logging with make LOG=1
ifeq ($(LOG),1)
  CFLAGS += -DENABLE_LOG
endif

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
	pytest -v tests
	rm -f stencil_parallel stencil_serial *.bin

# Clean built executables
clean:
	rm -f stencil_parallel stencil_serial *.bin

.PHONY: all clean
