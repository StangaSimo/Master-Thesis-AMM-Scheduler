TARGET = benchmark

SRC_DIR = src
SRC = $(SRC_DIR)/main.cpp

CXX = g++
CXXFLAGS = -std=c++17

LDFLAGS += -lopenvino

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
