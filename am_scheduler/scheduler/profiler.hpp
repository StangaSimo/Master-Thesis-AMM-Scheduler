#ifndef PROFILER_H
#define PROFILER_H

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <chrono>

#include "tasks.hpp"
#include "config.hpp"

#ifdef ENABLE_CUDA
#include <nvml.h>
#endif

/* total energy consumption in joules */
struct EnergyConsumption {
    double cpu_pkg_joules = 0.0;
    double cpu_core_joules = 0.0;
    double intel_gpu_joules = 0.0;
    double nvidia_gpu_joules = 0.0;
};

struct WorkerMetrics {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_work;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_work;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_work;

    double work = 0;
    double total_idle = 0;
    double total_work = 0;
    int tasks_processed = 0;

    void init_last_work() { last_work = std::chrono::high_resolution_clock::now(); }
    void start_worker() {
        start_work = std::chrono::high_resolution_clock::now();
        double idle_us = std::chrono::duration<double, std::micro>(start_work - last_work).count();
        total_idle += idle_us;
    }
    void end_worker() {
        end_work = std::chrono::high_resolution_clock::now();
        double work_us = std::chrono::duration<double, std::micro>(end_work - start_work).count();
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
        double res = std::chrono::duration<double, std::micro>(stop - start).count();
        sum += res;
        if (res < min) min = res;
        if (res > max) max = res;
        count++;
        return res;
    }
    void start_counter() { start = std::chrono::high_resolution_clock::now(); }
    void stop_counter() { stop = std::chrono::high_resolution_clock::now(); }
    double avg() const { return count > 0 ? sum / count : 0.0; }
};

class Profiler {
    Metric fetch{"Fetch"};
    Metric logic{"Logic"};
    Metric disp{"Dispatch"};

    std::unique_ptr<WorkerMetrics[]> workers;

    double total = 0;
    int batch_count = 0;

    bool power_monitoring_active = false;

    /* gpu accumulator */
    unsigned long long start_nvidia_mj = 0;

    /* rapl start */
    double start_pkg_uj = 0.0;
    double start_core_uj = 0.0;

    EnergyConsumption final_energy;

    public: 
    Profiler() : workers(new WorkerMetrics[BT::COUNT]) {}

    ~Profiler() {
        if(power_monitoring_active) stop_power_monitor();
    }

    void start_power_monitor() {
        if (power_monitoring_active) return;

        final_energy = {0,0,0};

       /* starting rapl read */
#ifdef ENABLE_INTEL_POWER_PROFILE
        start_pkg_uj = read_rapl_energy_uj("intel-rapl:0");
        start_core_uj = read_rapl_energy_uj("intel-rapl:0/intel-rapl:0:0");
        //std::cout << "\n[PROFILER] RAPL Start PKG: " << start_pkg_uj << " uJ\n";
#endif

       /* starting nvml read */
#ifdef ENABLE_CUDA
        nvmlReturn_t res = nvmlInit();
        if (res == NVML_SUCCESS) {
            nvmlDevice_t device;
            nvmlDeviceGetHandleByIndex(0, &device);
            unsigned long long energy = 0;
            if (nvmlDeviceGetTotalEnergyConsumption(device, &energy) == NVML_SUCCESS) {
                start_nvidia_mj = energy;
            } else {
                start_nvidia_mj = 0;
                std::cerr << "[PROFILER] Nvidia Energy Counter not supported on this device.\n";
                exit(EXIT_FAILURE);
            }
        }
#endif
        power_monitoring_active = true;
    }

    void stop_power_monitor() {
        if (!power_monitoring_active) return;

#ifdef ENABLE_INTEL_POWER_PROFILE
        double end_pkg_uj = read_rapl_energy_uj("intel-rapl:0");
        double end_core_uj = read_rapl_energy_uj("intel-rapl:0/intel-rapl:0:0");

        //std::cout << "[PROFILER] RAPL End PKG: " << end_pkg_uj << " uJ\n";

        if (end_pkg_uj >= start_pkg_uj)
            final_energy.cpu_pkg_joules = (end_pkg_uj - start_pkg_uj) / 1e6; // uJ to J
        
        if (end_core_uj >= start_core_uj)
            final_energy.cpu_core_joules = (end_core_uj - start_core_uj) / 1e6;
#endif

#ifdef ENABLE_CUDA
        unsigned long long end_nvidia_mj = 0;
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex(0, &device) == NVML_SUCCESS) {
            nvmlDeviceGetTotalEnergyConsumption(device, &end_nvidia_mj);
        }
        
        if (end_nvidia_mj >= start_nvidia_mj && start_nvidia_mj > 0) {
            final_energy.nvidia_gpu_joules = (double)(end_nvidia_mj - start_nvidia_mj) / 1000.0; // mJ to J
        }
        
        nvmlShutdown();
#endif

        power_monitoring_active = false;
    }

    void start_fetch() {fetch.start_counter();}
    void start_logic() {logic.start_counter();}
    void start_dispatch() {disp.start_counter();}
    void stop_fetch() {fetch.stop_counter();}
    void stop_logic() {logic.stop_counter();}
    void stop_dispatch() {disp.stop_counter();}

    void init_last_work(size_t backend_type) { workers[backend_type].init_last_work(); }
    void start_worker(size_t backend_type) { workers[backend_type].start_worker(); }
    void end_worker(size_t backend_type) { workers[backend_type].end_worker(); }

    void record_sample() {
        total += fetch.record() + logic.record() + disp.record();
        batch_count++;
    }

    void print_stats(const std::vector<BT>& bts, task* tasks, size_t num_tasks, std::string strategy_name = "UNKNOWN") {
        if (tasks == nullptr || num_tasks == 0) return;

        stop_power_monitor();

        double total_span_ms = 0, avg_lat_ms = 0, min_lat_ms = 0, max_lat_ms = 0;
        calculate_latency_metrics(tasks, num_tasks, total_span_ms, avg_lat_ms, min_lat_ms, max_lat_ms);

        print_latency_info(tasks[0].M, tasks[0].N, tasks[0].K, num_tasks, total_span_ms, avg_lat_ms, min_lat_ms, max_lat_ms);

        print_energy_stats(total_span_ms);

#ifdef ENABLE_PROFILING
        print_scheduler_metrics(bts);
#endif

        save_to_csv(strategy_name, tasks[0].M, tasks[0].N, tasks[0].K, num_tasks, 
                total_span_ms, avg_lat_ms, min_lat_ms, max_lat_ms, bts, final_energy);
    }

    void print_stats(const std::vector<BT>& bts, const std::vector<task*>& sub_tasks, std::string strategy_name = "UNKNOWN") {
        if (sub_tasks.empty()) return;
        std::vector<task> temp_tasks;
        temp_tasks.reserve(sub_tasks.size());
        for (auto* t : sub_tasks) temp_tasks.push_back(*t);
        print_stats(bts, temp_tasks.data(), temp_tasks.size(), strategy_name);
    }

    private:
    double read_rapl_energy_uj(const std::string &domain) {
        std::ifstream file("/sys/class/powercap/" + domain + "/energy_uj");
        double value = 0;
        if (file >> value) return value;
        return 0;
    }

    void calculate_latency_metrics(task* tasks, size_t num_tasks, double& total_span, double& avg, double& min, double& max) {
        if (num_tasks == 0) return;
        auto global_start = tasks[0].start_time;
        auto global_end = tasks[0].end_time;
        double sum = 0; min = std::numeric_limits<double>::max(); max = 0;

        for (size_t i=0; i<num_tasks; i++) {
            if (tasks[i].start_time < global_start) global_start = tasks[i].start_time;
            if (tasks[i].end_time > global_end) global_end = tasks[i].end_time;
            double curr = std::chrono::duration<double, std::milli>(tasks[i].end_time - tasks[i].start_time).count();
            sum += curr;
            if (curr < min) min = curr;
            if (curr > max) max = curr;
        }
        total_span = std::chrono::duration<double, std::milli>(global_end - global_start).count();
        avg = sum / num_tasks;
    }

    void print_latency_info(int M, int N, int K, int num, double total, double avg, double min, double max) {
        std::cout << "==================================\n";
        std::cout << "Matrix: " << M << "x" << N << "x" << K << " | Tasks: " << num << "\n";
        std::cout << "Total Time: " << total << " ms\n";
        std::cout << "Latency (ms): Avg " << avg << " | Min " << min << " | Max " << max << "\n";
    }

    void print_energy_stats(double total_ms) {
        double seconds = total_ms / 1000.0;
        if (seconds <= 0.0) seconds = 1.0;

        std::cout << "\nPower consumption: \nDuration: " << seconds << " s\n";
        std::cout << std::fixed << std::setprecision(2);

#ifdef ENABLE_INTEL_POWER_PROFILE
        double pkg_w = final_energy.cpu_pkg_joules / seconds;
        double core_w = final_energy.cpu_core_joules / seconds;

        std::cout << "cpu package: " << std::setw(8) << final_energy.cpu_pkg_joules << " J  (" << pkg_w << " W avg)\n";
        std::cout << "cpu core:    " << std::setw(8) << final_energy.cpu_core_joules << " J  (" << core_w << " W avg)\n";
#endif

#ifdef ENABLE_CUDA
        double nv_w = final_energy.nvidia_gpu_joules / seconds;
        std::cout << "cuda gpu:    " << std::setw(8) << final_energy.nvidia_gpu_joules << " J  (" << nv_w << " W avg)\n";
#endif
        double total_j = final_energy.cpu_pkg_joules + final_energy.nvidia_gpu_joules;
        std::cout << "total:    " << std::setw(8) << total_j << " J\n";

    }

    void print_scheduler_metrics(const std::vector<BT>& bts) {
        if (batch_count == 0) return;

        std::cout << "\nScheduler: " << batch_count << " batches\n";
        print_metric(fetch);
        print_metric(logic);
        print_metric(disp);
        std::cout << std::left << std::setw(10) << "Total avg overhead: " << total/batch_count << " us  Total overhead: " << total << " us\n";

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

            std::string acc_str = get_acc_string(i);
            std::cout << "[" << acc_str << "]" << "\n";
            std::cout << "  Tasks: " << w.tasks_processed << "\n";
            std::cout << "  Occupancy:         " << std::fixed << std::setprecision(2) << occupancy << " %\n";
            std::cout << "  Total idle time: " << total_idle_ms << " ms,  Avg idle time:   " << avg_idle_us << " us\n";
            std::cout << "  Total work time: " << total_work_ms << " ms,  Avg work time:   " << avg_work_us << " us\n";
            std::cout << "  total time:       " << total_time_ms << " ms\n";
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
            const std::vector<BT>& bts, const EnergyConsumption& en) {

        std::string filename = "bin/csv/results_full.csv";

        std::filesystem::path path(filename);
        if (path.has_parent_path() && !std::filesystem::exists(path.parent_path())) {
            std::filesystem::create_directories(path.parent_path());
        }

        bool file_exists = std::filesystem::exists(filename);
        std::ofstream file(filename, std::ios::app);/* append mode */

        if (!file.is_open()) {
            std::cerr << "[PROFILER] Error opening CSV file: " << filename << std::endl;
            return;
        }

        double seconds = total_ms / 1000.0;
        if (seconds <= 0) seconds = 1.0;

        double pkg_w = en.cpu_pkg_joules / seconds;
        double core_w = en.cpu_core_joules / seconds;
        double nv_w = en.nvidia_gpu_joules / seconds;

        if (!file_exists || std::filesystem::file_size(filename) == 0) {
            /* global state */
            file << "Strategy,M,N,K,Num_Tasks,Total_Time_ms,Total_Time_s,Avg_Lat_ms,Min_Lat_ms,Max_Lat_ms";

            /* power stats */
            file << ",Pkg_J,Pkg_Avg_W,Core_J,Core_Avg_W,NVGPU_J,NVGPU_Avg_W";

            /* scheduler metrics */
            file << ",Sched_Avg_Fetch_us,Sched_Avg_Logic_us,Sched_Avg_Disp_us";

            /* worker metrics */
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
            << total_ms << "," << std::fixed << std::setprecision(3) << seconds << "," 
            << avg_ms << "," << min_ms << "," << max_ms;

        /* power stats */
#ifdef ENABLE_INTEL_POWER_PROFILE
        file << "," << en.cpu_pkg_joules << "," << pkg_w
            << "," << en.cpu_core_joules << "," << core_w;
#else
        file << ",0,0,0,0";
#endif

#ifdef ENABLE_CUDA
        file << "," << en.nvidia_gpu_joules << "," << nv_w;
#else
        file << ",0,0";
#endif

        /* scheduler metrics */
        file << "," << fetch.avg() << "," << logic.avg() << "," << disp.avg();

        /* workers metrics */
        for (int i = 0; i < BT::COUNT; ++i) {
            if (i == BT::CORDINATOR) {continue;}
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
                file << ",0,0,0,0,0,0,0";
            }
        }
        file << "\n";
        file.close();
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
