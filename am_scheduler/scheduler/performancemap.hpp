#ifndef REGRESSION_H
#define REGRESSION_H

#include "regression.hpp"
#include <iostream>
#include <unordered_map>
#include <fstream>
#include <vector>
#include <string>

#include <cstdlib>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

inline vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);

    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);

    }
    return tokens;
}

/* 1048576 limit for each M or N or K */
class PerformanceMap {

    unordered_map<unsigned long long, double> grid_map;
    long long STEP_SIZE;

    long long last_M = 0;    
    long long last_N = 0;    
    long long last_K = 0;    
    double last_key = 0.0;    

private:

    /* return the successor of val in our step_size */
    long long round_up(long long val) {
        if (val == 0) return STEP_SIZE;
        return ((val + STEP_SIZE - 1) / STEP_SIZE) * STEP_SIZE;
    }

    /* return the key of val in our step_size */
    unsigned long long get_key(long long M, long long N, long long K) {
        long long rM = round_up(M);
        long long rN = round_up(N);
        long long rK = round_up(K);

        /* 64 bit  |4bit nil|20bit M|20bit N|20bit K| */
        return ((unsigned long long)rM << 40) | ((unsigned long long)rN << 20) | (unsigned long long)rK;
    }

    void add_key(long long M, long long N, long long K, double time) {
        unsigned long long key = get_key(M, N, K);
        grid_map[key] = time;
    }

    /* open file and init the map */
    void init_map(string filename){
        ifstream file(filename);

        if (!file.is_open()) {
            cerr << "[PerformanceMap]: ERROR open file" << filename <<" \n";
            exit(EXIT_FAILURE);
        }

        string line;
        getline(file, line);
        while (std::getline(file, line)) {
            vector<std::string> row = split(line, ',');
            if (row.size() < 6) continue;

            int M = stod(row[2]); 
            int N = stod(row[3]);
            int K = stod(row[4]);
            double time = stod(row[5]);
            
            add_key(M, N, K, time);
        }
    }

public:

    PerformanceMap(int step_size, string filename) : STEP_SIZE(step_size) {
        init_map(filename);        
    }

    /* return the time, -1 if the matrix is too big */
    double query(long long M, long long N, long long K) {
        unsigned long long key = get_key(M, N, K);

        /* we cache the last results */
        if (M == last_M && N == last_N && K == last_K) {return last_K;}

        last_M = M;
        last_N = N;
        last_K = K;

        if (grid_map.find(key) != grid_map.end()) {
            last_K = grid_map[key];
            return grid_map[key];
        }

        last_K = -1.0;
        return -1.0; 
    }
};

#endif
