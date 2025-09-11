NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17
TARGET = gemm_benchmark
SRC = src/gemm.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean run
