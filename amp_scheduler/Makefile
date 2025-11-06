TARGET = mio_eseguibile

BUILD_DIR = build

EXECUTABLE = $(BUILD_DIR)/$(TARGET)

all: $(EXECUTABLE)

$(EXECUTABLE): CMakeLists.txt main.cpp cuda_hello.cu
	@echo "--- [BUILD] Configurazione CMake in corso... ---"
	# Crea la cartella di build e lancia cmake
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ..
	
	@echo "--- [BUILD] Compilazione in corso... ---"
	# Lancia 'make' dentro la cartella di build
	$(MAKE) -C $(BUILD_DIR)

run: all
	@echo "--- [RUN] Esecuzione di $(TARGET) ---"
	./$(EXECUTABLE)

clean:
	@echo "--- [CLEAN] Pulizia di $(BUILD_DIR) ---"
	rm -rf $(BUILD_DIR)

.PHONY: all run clean