#ifndef PROFILER_H
#define PROFILER_H

#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <algorithm> // per std::find
#include "tasks.hpp"
#include "config.hpp" // Assicurati che get_acc_string sia qui o accessibile

struct WorkerMetrics {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_work;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_work;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_work;
    
    // Accumulatori in microsecondi (us)
    double work = 0;
    double total_idle = 0;
    double total_work = 0;
    int tasks_processed = 0;

    void init_last_work() {
        last_work = chrono::high_resolution_clock::now();
    }

    void start_worker() {
        start_work = chrono::high_resolution_clock::now();
        double idle_us = chrono::duration<double, micro>(start_work - last_work).count();
        total_idle += idle_us;
    }

    void end_worker() {
        end_work = chrono::high_resolution_clock::now();
        double work_us = chrono::duration<double, micro>(end_work - start_work).count();
        total_work += work_us;
        tasks_processed++;
        last_work = end_work;
    }
};

struct Metric {
    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop;
    double sum = 0.0;
    double min = std::numeric_limits<double>::max();
    double max = 0.0;
    int count = 0;

    double record() {
        double res = chrono::duration<double, micro>(stop - start).count();
        sum += res;
        if (res < min) min = res;
        if (res > max) max = res;
        count++;
        return res;
    }

    void start_counter() {
        start = chrono::high_resolution_clock::now();
    }

    void stop_counter() {
        stop = chrono::high_resolution_clock::now();
    }

    double avg() const { 
        return count > 0 ? sum / count : 0.0; 
    }

    void reset() {
        sum = 0.0; 
        min = std::numeric_limits<double>::max(); 
        max = 0.0; 
        count = 0;
    }
};


class Profiler {

    Metric fetch{"Fetch"};
    Metric logic{"Logic"};
    Metric disp{"Dispatch"};

    unique_ptr<WorkerMetrics[]> workers;

    double total = 0;
    int batch_count = 0;

public: 
    Profiler() : workers(new WorkerMetrics[BT::COUNT]) {
    }

    void start_fetch() {fetch.start_counter();}
    void start_logic() {logic.start_counter();}
    void start_dispatch() {disp.start_counter();}
    void stop_fetch() {fetch.stop_counter();}
    void stop_logic() {logic.stop_counter();}
    void stop_dispatch() {disp.stop_counter();}

    void init_last_work(size_t backend_type) {
        workers[backend_type].init_last_work();
    }

    void start_worker(size_t backend_type) {
        workers[backend_type].start_worker();
    }
    void end_worker(size_t backend_type) {
        workers[backend_type].end_worker();
    }

    void record_sample() {
        double fetch_us = fetch.record();
        double logic_us = logic.record();
        double disp_us = disp.record();
        total += fetch_us + logic_us + disp_us;
        batch_count++;
    }

    void print_stats(const vector<BT>& bts, task* tasks, size_t num_tasks, std::string strategy_name = "UNKNOWN") {
        if (tasks == nullptr || num_tasks == 0) return;

        double total_span_ms = 0;
        double avg_lat_ms = 0;
        double min_lat_ms = 0;
        double max_lat_ms = 0;
        
        calculate_latency_metrics(tasks, num_tasks, total_span_ms, avg_lat_ms, min_lat_ms, max_lat_ms);

        print_latency_info(tasks[0].M, tasks[0].N, tasks[0].K, num_tasks, total_span_ms, avg_lat_ms, min_lat_ms, max_lat_ms);

        #ifdef ENABLE_PROFILING
        print_scheduler_metrics(bts);
        #endif

        save_to_csv(strategy_name, tasks[0].M, tasks[0].N, tasks[0].K, num_tasks, 
                    total_span_ms, avg_lat_ms, min_lat_ms, max_lat_ms, bts);
    }

    /* large matrix multiplication call this function */
    void print_stats(const vector<BT>& bts, const vector<task*>& sub_tasks, std::string strategy_name = "UNKNOWN") {
        if (sub_tasks.empty()) return;
        
        vector<task> temp_tasks;
        temp_tasks.reserve(sub_tasks.size());
        for (auto* t : sub_tasks) {
            temp_tasks.push_back(*t);
        }

        print_stats(bts, temp_tasks.data(), temp_tasks.size(), strategy_name);
    }

private:
    void calculate_latency_metrics(task* tasks, size_t num_tasks, double& total_span, double& avg, double& min, double& max) {
        auto global_start = tasks[0].start_time;
        auto global_end = tasks[0].end_time;
        
        double total_latency_sum = 0.0;
        min = std::numeric_limits<double>::max();
        max = 0.0;

        for (size_t i=0; i<num_tasks; i++) {
            if (tasks[i].start_time < global_start) global_start = tasks[i].start_time;
            if (tasks[i].end_time > global_end) global_end = tasks[i].end_time;
            
            double curr_ms = std::chrono::duration<double, std::milli>(tasks[i].end_time - tasks[i].start_time).count();
            if (curr_ms < 0) continue;

            total_latency_sum += curr_ms;
            if (curr_ms < min) min = curr_ms;
            if (curr_ms > max) max = curr_ms;
        }

        total_span = std::chrono::duration<double, std::milli>(global_end - global_start).count();
        avg = (num_tasks > 0) ? total_latency_sum / num_tasks : 0.0;
    }

    void print_latency_info(int M, int N, int K, int num, double total, double avg, double min, double max) {
        std::cout << "==================================\n";
        std::cout << "N Matrix: " << num << "\n";
        std::cout << "M : " << M  << " N : " << N << " K : " << K << "\n";
        std::cout << "\nAvg latency:      " << avg << " ms\n";
        std::cout << "Min latency:      " << min << " ms\n";
        std::cout << "Max latency:      " << max << " ms\n";
        std::cout << "\nTotal ms:        " << total << " ms\n";
        std::cout << "\n";
    }
   
    void print_scheduler_metrics(const vector<BT>& bts) {
        if (batch_count == 0) return;

        std::cout << "Scheduler: " << batch_count << " batches\n";
        print_metric(fetch);
        print_metric(logic);
        print_metric(disp);
        std::cout << std::left << std::setw(10) << "Total overhead: " << total/batch_count << " us\n";

        std::cout << "\nWorkers:\n";

        for (BT i : bts) {
            const auto& w = workers[i];
            
            double total_time_us = w.total_work + w.total_idle;
            double occupancy = (total_time_us > 0) ? (w.total_work / total_time_us) * 100.0 : 0.0;
            
            double avg_idle_us = (w.tasks_processed > 0) ? w.total_idle / w.tasks_processed : 0.0;
            double avg_work_us = (w.tasks_processed > 0) ? w.total_work / w.tasks_processed : 0.0;

            double total_idle_ms = w.total_idle / 1000.0;
            double total_work_ms = w.total_work / 1000.0;
            double total_time_ms = total_time_us / 1000.0;

            string acc_str = get_acc_string(i);
            std::cout << "[" << acc_str << "]" << "\n";
            std::cout << "  Tasks: " << w.tasks_processed << "\n";
            std::cout << "  Occupancy:         " << std::fixed << std::setprecision(2) << occupancy << " %\n";
            std::cout << "  Total idle time: " << total_idle_ms << " ms,  Avg idle time:   " << avg_idle_us << " us\n";
            std::cout << "  Total work time: " << total_work_ms << " ms,  Avg work time:   " << avg_work_us << " us\n";
            std::cout << "  total time:      " << total_time_ms << " ms\n";
        }
    }

    void print_metric(const Metric& m) {
        std::cout << std::left << std::setw(10) << m.name 
                  << " | Avg: " << std::setw(6) << std::fixed << std::setprecision(2) << m.avg() << " us"
                  << " | Min: " << std::setw(6) << m.min << " us"
                  << " | Max: " << std::setw(6) << m.max << " us\n";
    }

    void save_to_csv(std::string strategy, int M, int N, int K, int num_tasks, 
                     double total_ms, double avg_ms, double min_ms, double max_ms, 
                     const vector<BT>& bts) {
        std::string filename = "bin/csv/results_full.csv";
        
        std::filesystem::path path(filename);
        if (path.has_parent_path() && !std::filesystem::exists(path.parent_path())) {
            std::filesystem::create_directories(path.parent_path());
        }

        bool file_exists = std::filesystem::exists(filename);
        std::ofstream file(filename, std::ios::app); // Append mode

        if (!file.is_open()) {
            std::cerr << "[PROFILER] Error opening CSV file: " << filename << std::endl;
            return;
        }

        // Header FISSO: ciclo su tutti i possibili backend (0 .. BT::COUNT)
        if (!file_exists || std::filesystem::file_size(filename) == 0) {
            file << "Strategy,M,N,K,Num_Tasks,Total_Time_ms,Avg_Lat_ms,Min_Lat_ms,Max_Lat_ms,"
                 << "Sched_Avg_Fetch_us,Sched_Avg_Logic_us,Sched_Avg_Disp_us";
            
            for (int i = 0; i < BT::COUNT; ++i) {
                if (i == BT::CORDINATOR) {continue;}
                std::string name = get_acc_string((BT)i);
                file << "," << name << "_Tasks"
                     << "," << name << "_Occupancy_%"
                     << "," << name << "_Tot_Idle_ms"
                     << "," << name << "_Tot_Work_ms"
                     << "," << name << "_Avg_Idle_us"
                     << "," << name << "_Avg_Work_us"
                     << "," << name << "_Tot_Time_ms";
            }
            file << "\n";
        }

        /* global stats */
        file << strategy << "," << M << "," << N << "," << K << "," << num_tasks << "," 
             << total_ms << "," << avg_ms << "," << min_ms << "," << max_ms;

        /* scheduler */
        file << "," << fetch.avg() << "," << logic.avg() << "," << disp.avg();

        /* workers */
        for (int i = 0; i < BT::COUNT; ++i) {
            if (i == BT::CORDINATOR) {continue;}
            // Controlla se il backend 'i' è presente nel vettore 'bts' (è attivo?)
            bool is_active = false;
            for(auto b : bts) { if((int)b == i) is_active = true; }

            if (is_active) {
                const auto& w = workers[i];
                
                double total_time_us = w.total_work + w.total_idle;
                double occupancy = (total_time_us > 0) ? (w.total_work / total_time_us) * 100.0 : 0.0;
                
                double avg_idle_us = (w.tasks_processed > 0) ? w.total_idle / w.tasks_processed : 0.0;
                double avg_work_us = (w.tasks_processed > 0) ? w.total_work / w.tasks_processed : 0.0;

                double total_idle_ms = w.total_idle / 1000.0;
                double total_work_ms = w.total_work / 1000.0;
                double total_time_ms = total_time_us / 1000.0;

                file << "," << w.tasks_processed
                     << "," << occupancy
                     << "," << total_idle_ms
                     << "," << total_work_ms
                     << "," << avg_idle_us
                     << "," << avg_work_us
                     << "," << total_time_ms;
            } else {
                /* backend not active */
                file << ",-1,-1,-1,-1,-1,-1,-1";
            }
        }
        file << "\n";
        
        file.close();
        //std::cout << "[PROFILER] Full stats saved to " << filename << "\n";
    }
};

inline void print_performance_stats(task* tasks, size_t num_tasks) {

    if (tasks == nullptr || num_tasks == 0) {
        std::cout << "[ERROR] print_performance_stats\n";
        exit(EXIT_FAILURE);
    }

    auto min_start = tasks[0].start_time;
    auto max_end = tasks[0].end_time;
    int M = tasks[0].M;
    int N = tasks[0].N;
    int K = tasks[0].K;

    std::chrono::duration<double, std::milli> first_time = tasks[0].end_time - tasks[0].start_time;
    double min_latency_ms = first_time.count();
    double max_latency_ms = first_time.count();
    double total_latency_sum_ms = 0.0;

    for (size_t i=0; i<num_tasks; i++) {

        if (tasks[i].start_time < min_start) 
            min_start = tasks[i].start_time;

        if (tasks[i].end_time > max_end) 
            max_end = tasks[i].end_time;

        std::chrono::duration<double, std::milli> duration = tasks[i].end_time - tasks[i].start_time;
        double curr_latency_ms = duration.count();

        total_latency_sum_ms += curr_latency_ms;

        if (curr_latency_ms < min_latency_ms) 
            min_latency_ms = curr_latency_ms;

        if (curr_latency_ms > max_latency_ms) 
            max_latency_ms = curr_latency_ms;
    }

    std::chrono::duration<double, std::milli> global_span = max_end - min_start;
    double average_latency_ms = total_latency_sum_ms / num_tasks;
    std::cout << "==================================\n";
    std::cout << "N Matrix: " << num_tasks << "\n";
    std::cout << "M : " << M  << " N : " << N << " K : " << K << "\n";
    std::cout << "\nAvg latency:      " << average_latency_ms << " ms\n";
    std::cout << "Min latency:      " << min_latency_ms << " ms\n";
    std::cout << "Max latency:      " << max_latency_ms << " ms\n";
    std::cout << "\nTotal ms:       " << global_span.count() << " ms\n";
    std::cout << "\n";
}

#endif
