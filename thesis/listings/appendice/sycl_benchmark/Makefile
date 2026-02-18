TARGET = sycl_benchmark
BUILD_DIR = build
EXECUTABLE = $(BUILD_DIR)/$(TARGET)
CXX = icpx

SRC_FILES = $(wildcard *.cpp)

all: $(EXECUTABLE)

$(EXECUTABLE): CMakeLists.txt $(SRC_FILES)
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake -DCMAKE_CXX_COMPILER=$(CXX) ..
	
	$(MAKE) -C $(BUILD_DIR)

run: all
	sudo ./$(EXECUTABLE)

clean:
	rm -rf $(BUILD_DIR)

fresh: clean run

.PHONY: all run clean fresh
