NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17
TARGET = benchmark

# Directory structure
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIRS = -I$(SRC_DIR)/include

# Source files
SRC = $(SRC_DIR)/main.cu \
      $(SRC_DIR)/benchmarks/gem_runner.cu \
      $(SRC_DIR)/benchmarks/mem_runner.cu \
      $(SRC_DIR)/kernels/simple_gem.cu \
      $(SRC_DIR)/kernels/simple_mem.cu \
      $(SRC_DIR)/kernels/title_gem.cu \
      $(SRC_DIR)/kernels/dense_gem.cu

OBJ = $(SRC:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -o $@ $(OBJ)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(@D)  # Create directory structure in build
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: clean run