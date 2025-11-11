# --- Configurazione del Target ---
TARGET = benchmark

# --- Sorgenti ---
# Come hai richiesto, il main è in 'src'
SRC_DIR = src
SRC = $(SRC_DIR)/main.cpp

# --- Compilatore e Flag ---
CXX = g++

# Flag di base: C++17 e Ottimizzazione -O3 (fondamentale per i benchmark)
CXXFLAGS = -std=c++17 -O3

# Flag per OpenCV:
# $(shell ...) esegue il comando pkg-config e ne inserisce l'output
# CXXFLAGS ottiene i percorsi -I (include)
# LDFLAGS ottiene i percorsi -L e le librerie -l (link)
CXXFLAGS += $(shell pkg-config opencv4 --cflags)
LDFLAGS   = $(shell pkg-config opencv4 --libs)


# --- Regole (Targets) ---

# .PHONY dice a 'make' che queste non sono nomi di file
# 'all' è il target di default
.PHONY: all clean run

all: $(TARGET)

# Regola per costruire il target
# Nota: Ho aggiunto $(CXXFLAGS) alla regola di compilazione,
# che mancava nel tuo esempio ma è essenziale per usare gli include.
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

# 'make clean' - Pulisce i file compilati
clean:
	rm -f $(TARGET)

# 'make run' - Compila (se necessario) ed esegue
run: all
	./$(TARGET)