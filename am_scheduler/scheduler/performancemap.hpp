#ifndef PERFORMANCEMAP_H
#define PERFORMANCEMAP_H

#include "tasks.hpp"
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

    struct map_struct {
        unordered_map<unsigned long long, double> grid_map;
        
        long long last_M = -1;    
        long long last_N = -1;    
        long long last_K = -1;    
        double last_result = -1.0; 
    };

    map_struct map_f;
    map_struct map_h;

    long long STEP_SIZE;

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

    void add_key(long long M, long long N, long long K, double time, Type type) {
        unsigned long long key = get_key(M, N, K);
        if (type == Type::FLOAT)
           map_f.grid_map[key] = time;
        if (type == Type::HALF)
           map_h.grid_map[key] = time;
    }
    
    void add_keys(ifstream* file, Type type) {
        string line;
        getline(*file, line);
        while (std::getline(*file, line)) {
            vector<std::string> row = split(line, ',');
            if (row.size() < 6) continue;

            int M = stod(row[2]); 
            int N = stod(row[3]);
            int K = stod(row[4]);
            double time = stod(row[5]);
            
            add_key(M, N, K, time, type);
        }
    }

    /* open file and init the map */
    void init_maps(string filename_f, string filename_h){
        ifstream file_f(filename_f);
        ifstream file_h(filename_h);

        if (!file_f.is_open()) {
            cerr << "[PerformanceMap]: ERROR open file" << filename_f <<" \n";
            exit(EXIT_FAILURE);
        }

        if (!file_h.is_open()) {
            cerr << "[PerformanceMap]: ERROR open file" << filename_h <<" \n";
            exit(EXIT_FAILURE);
        }

        add_keys(&file_f,Type::FLOAT);
        add_keys(&file_h,Type::HALF);
    }

public:

    PerformanceMap(int step_size, string filename_f, string filename_h) : STEP_SIZE(step_size) {
        init_maps(filename_f, filename_h);        
    }

    /* return the time, -1 if the matrix is too big */
    double query(long long M, long long N, long long K, Type type) {
        map_struct &data = (type == Type::FLOAT) ? map_f : map_h;

        unsigned long long key = get_key(M, N, K);

        /* we cache the last results */
        if (M == data.last_M && N == data.last_N && K == data.last_K) {return data.last_K;}

        data.last_M = M;
        data.last_N = N;
        data.last_K = K;

        if (data.grid_map.find(key) != data.grid_map.end()) {
            data.last_K = data.grid_map[key];
            return data.grid_map[key];
        }

        data.last_K = -1.0;
        return -1.0; 
    }
};

#endif
