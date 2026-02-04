#ifndef PERFORMANCEMAP_H
#define PERFORMANCEMAP_H

#include "tasks.hpp"
#include "config.hpp"

#include <iostream>
#include <tuple>
#include <unordered_map>
#include <unordered_set> /* for jit times */
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
        unordered_map<unsigned long long, tuple<double, double>> grid_map;
        
        long long last_M = -1;    
        long long last_N = -1;    
        long long last_K = -1;    
        double last_result = -1.0; 
        double max_result = -1.0;
    };

    map_struct map_f;
    map_struct map_h;

    unordered_set<unsigned long long> jit_cache_f;
    unordered_set<unsigned long long> jit_cache_h;

    /* which accellerator am i */
    BT bt;

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

    void add_key(long long M, long long N, long long K, double time, double jit_time, Type type) {
        unsigned long long key = get_key(M, N, K);
        if (type == Type::FLOAT){
           map_f.grid_map[key] = std::make_tuple(time, jit_time);
           map_f.max_result = (map_f.max_result > time ) ? map_f.max_result : time;
        }
        if (type == Type::HALF){
           map_h.grid_map[key] = std::make_tuple(time, jit_time);
           map_h.max_result = (map_h.max_result > time ) ? map_h.max_result : time;
        }
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
            double jit_time = stod(row[6]);
            
            add_key(M, N, K, time, jit_time, type);
        }
    }

    /* open file and init the map */
    void init_maps(string filename_f, string filename_h) {
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

    PerformanceMap(int step_size, BT bt, string filename_f, string filename_h) : STEP_SIZE(step_size), bt(bt) {
        init_maps(filename_f, filename_h);        
    }

    /* return the time, add_jit true = populate the jit cache */
    double query(long long M, long long N, long long K, Type type, bool add_jit) {
        map_struct* data = nullptr;
        unordered_set<unsigned long long>* jit_cache = nullptr;
        double time_ms, jit_ms;

        unsigned long long key = get_key(M, N, K);

        if (type == Type::FLOAT) {
            jit_cache = &jit_cache_f;
            data = &map_f;
        } else {
            jit_cache = &jit_cache_h;
            data = &map_h;
        }

        /* we cache the last results */
        if (M == data->last_M && N == data->last_N && K == data->last_K) {return data->last_K;}

        data->last_M = M;
        data->last_N = N;
        data->last_K = K;

        /* return if we find the key */
        if (data->grid_map.find(key) != data->grid_map.end()) {
            time_ms = get<0>(data->grid_map[key]);
            jit_ms = get<1>(data->grid_map[key]);

            /* add jit_ms only if openvino never see M N K*/
            if (bt == BT::OPENVINO) {

                /* mitigate overhead when other accellerator are running */
                time_ms += time_ms * 0.1;

                if (jit_cache->find(key) == jit_cache->end()){
                    if (add_jit)
                        jit_cache->insert(key);

                    time_ms += jit_ms * 1.3;
                    //time_ms += jit_ms;

                }
            }

            /* add jit_ms only if sycl never see N */
            if (bt == BT::SYCL) {

                /* mitigate overhead when other accellerator are running */
                //time_ms += time_ms * 0.5;

                unsigned long long sycl_key = (unsigned long long) N;
                if (jit_cache->find(sycl_key) == jit_cache->end()){

                    if (add_jit)
                        jit_cache->insert(sycl_key);

                    time_ms += JIT_MS_SYCL;
                }
            }
            
            /* mitigate openblas error in bencharks  */
            if (bt == BT::OPENBLAS)
                time_ms = time_ms * 0.5;
            
            data->last_K = time_ms;
            return time_ms;
        }

        /* if we don't have data on this matrix, just return max time * 2 */
        time_ms = data->max_result * 2;

        /* add jit expences */
        if (bt == BT::SYCL || bt == BT::OPENVINO) {
            time_ms += JIT_MS_SYCL;
        }

        /* cache the result */
        data->last_K = time_ms;

        return time_ms; 
    }
};
#endif
