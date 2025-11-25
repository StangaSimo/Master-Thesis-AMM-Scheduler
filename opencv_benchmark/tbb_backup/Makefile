TARGET = benchmark

SRC_DIR = src
SRC = $(SRC_DIR)/main.cpp

CXX = g++
CXXFLAGS = -std=c++17 -O3 -pthread

CXXFLAGS += $(shell pkg-config opencv4 --cflags)
LDFLAGS   = $(shell pkg-config opencv4 --libs)

CXXFLAGS += $(shell pkg-config tbb --cflags)
LDFLAGS  += $(shell pkg-config tbb --libs)

CXXFLAGS += $(shell pkg-config mkl-dynamic-lp64-seq --cflags)
LDFLAGS  += $(shell pkg-config mkl-dynamic-lp64-seq --libs)

LDFLAGS  += -lpthread -lm -ldl

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)