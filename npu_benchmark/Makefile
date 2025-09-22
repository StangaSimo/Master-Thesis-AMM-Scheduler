TARGET = benchmark

SRC_DIR = src
SRC = $(SRC_DIR)/main.cpp

CXX = g++
CXXFLAGS = -std=c++17

ifndef OPENVINO_DIR
$(error Please set the OPENVINO_DIR environment variable to your OpenVINO build directory)
endif

CXXFLAGS += -I$(OPENVINO_DIR)/runtime/include
LDFLAGS += -L$(OPENVINO_DIR)/runtime/lib/intel64 -lopenvino

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
