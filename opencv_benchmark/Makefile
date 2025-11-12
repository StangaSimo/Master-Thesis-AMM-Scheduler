# --- Configurazione del Target ---
TARGET = benchmark

# --- Sorgenti ---
SRC_DIR = src
SRC = $(SRC_DIR)/main.cpp

# --- Compilatore e Flag ---
CXX = g++
CXXFLAGS = -std=c++17 -O3 -pthread

# --- Flag di OpenCV ---
CXXFLAGS += $(shell pkg-config opencv4 --cflags)
LDFLAGS   = $(shell pkg-config opencv4 --libs)

# --- Flag di TBB (Sappiamo che questo funziona) ---
# pkg-config tbb --cflags non restituisce nulla, va bene
CXXFLAGS += $(shell pkg-config tbb --cflags)
# pkg-config tbb --libs aggiunge -ltbb
LDFLAGS  += $(shell pkg-config tbb --libs)

# --- Flag di oneMKL (Link MANUALE) ---
#
# Visto che 'pkg-config mkl-...' non funziona,
# linkiamo manualmente le librerie TBB-threaded di MKL.
# L'ordine è importante!
#
# -lmkl_intel_lp64:   Interfaccia LP64 (interi a 64-bit)
# -lmkl_tbb_thread:   Layer di threading basato su TBB
# -lmkl_core:         Il "cuore" computazionale di MKL
#
LDFLAGS  += -lmkl_intel_lp64 -lmkl_tbb_thread -lmkl_core

# Librerie di supporto standard (già presenti, ma per sicurezza)
LDFLAGS  += -lpthread -lm -ldl

# --- Regole (Targets) ---

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: all
	./$(TARGET)