TARGET = sycl_benchmark
BUILD_DIR = build
EXECUTABLE = $(BUILD_DIR)/$(TARGET)
CXX = icpx

SRC_FILES = $(wildcard *.cpp)

all: $(EXECUTABLE)

$(EXECUTABLE): CMakeLists.txt $(SRC_FILES)
	@echo "--- [BUILD] Configurazione CMake in corso... ---"
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_CXX_COMPILER=$(CXX) ..
	
	@echo "--- [BUILD] Compilazione in corso... ---"
	$(MAKE) -C $(BUILD_DIR)

run: all
	@echo "--- [RUN] Esecuzione di $(TARGET) ---"
	sudo ./$(EXECUTABLE)

clean:
	@echo "--- [CLEAN] Pulizia di $(BUILD_DIR) ---"
	rm -rf $(BUILD_DIR)

fresh: clean run

.PHONY: all run clean fresh
