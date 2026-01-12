#ifndef PROFILER_H
#define PROFILER_H

#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include "tasks.hpp"

struct WorkerMetrics {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_work;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_work;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_work;
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
        double work = chrono::duration<double, micro>(end_work - start_work).count();
        total_work += work;
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
    int count = 0;

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
        count++;
    }

    void print_stats(vector<BT> bts) {
        if (count == 0) return;

        std::cout << "Scheduler: " << count << " batches\n";
        print_metric(fetch);
        print_metric(logic);
        print_metric(disp);
        std::cout << std::left << std::setw(10) << "Total time: " << total/count << "us\n";

        std::cout << "\nWorkers:\n";

        for (BT i : bts) {
            const auto& w = workers[i];
            double total_time = w.total_work + w.total_idle;
            double occupancy = (total_time > 0) ? (w.total_work / total_time) * 100.0 : 0.0;
            double avg_idle = (w.tasks_processed > 0) ? w.total_idle / w.tasks_processed : 0.0;
            double avg_work = (w.tasks_processed > 0) ? w.total_work / w.tasks_processed : 0.0;

            string acc_str = get_acc_string(i);
            std::cout << "[" << acc_str << "]" << "\n";
            std::cout << "  Tasks: " << w.tasks_processed << "\n";
            std::cout << "  Occupancy:       " << std::fixed << std::setprecision(2) << occupancy << " %\n";
            std::cout << "  Total idle time: " << w.total_idle << " us,  Avg idle time:   " << avg_idle << " us\n";
            std::cout << "  Total work time: " <<  w.total_work << " us,  Avg work time:   " << avg_work << " us\n";
            std::cout << "  total time:   " << total_time/1000 << " ms\n";
        }

        std::cout << "===================================\n";
    }

private:
    void print_metric(const Metric& m) {
        std::cout << std::left << std::setw(10) << m.name 
                  << " | Avg: " << std::setw(3) << std::fixed << std::setprecision(2) << m.avg() << " us"
                  << " | Min: " << std::setw(3) << m.min << " us"
                  << " | Max: " << std::setw(3) << m.max << " us\n";
    }
};

/* it prints the performance stats of the tasks */
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

    for (size_t i=1; i<num_tasks; i++) {

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
    std::cout << "\nTotal ms:      " << global_span.count() << " ms\n";
    std::cout << "\n";
}
#endif
