#pragma once
#include <nvml.h>
#include <thread>
#include <atomic>
#include <vector>
#include <numeric>
#include <chrono>
#include <mutex>
#include <iostream>
#include <limits>

class GpuPowerSampler {
public:
    GpuPowerSampler(unsigned int deviceIndex = 0, int sampleIntervalMs = 100)
        : deviceIndex_(deviceIndex),
          sampleIntervalMs_(sampleIntervalMs),
          running_(false),
          minPower_(std::numeric_limits<double>::max()),
          maxPower_(0.0)
    {
        nvmlReturn_t result = nvmlInit();
        if (result != NVML_SUCCESS) {
            std::cerr << "NVML init failed: " << nvmlErrorString(result) << std::endl;
        }

        result = nvmlDeviceGetHandleByIndex(deviceIndex_, &device_);
        if (result != NVML_SUCCESS) {
            std::cerr << "NVML device get failed: " << nvmlErrorString(result) << std::endl;
        }
    }

    ~GpuPowerSampler() {
        stop();
        nvmlShutdown();
    }

    void start() {
        if (running_) return;
        running_ = true;
        samplerThread_ = std::thread(&GpuPowerSampler::sampleLoop, this);
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (samplerThread_.joinable())
            samplerThread_.join();
    }

    double averagePower() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (samples_.empty()) return 0.0;
        double sum = std::accumulate(samples_.begin(), samples_.end(), 0.0);
        return sum / samples_.size();
    }

    double minPower() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return samples_.empty() ? 0.0 : minPower_;
    }

    double maxPower() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return samples_.empty() ? 0.0 : maxPower_;
    }

    const std::vector<double>& samples() const { return samples_; }

private:
    void sampleLoop() {
        while (running_) {
            unsigned int power_mW = 0;
            if (nvmlDeviceGetPowerUsage(device_, &power_mW) == NVML_SUCCESS) {
                double power_W = power_mW / 1000.0;

                std::lock_guard<std::mutex> lock(mutex_);
                samples_.push_back(power_W);
                if (power_W < minPower_) minPower_ = power_W;
                if (power_W > maxPower_) maxPower_ = power_W;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(sampleIntervalMs_));
        }
    }

    unsigned int deviceIndex_;
    int sampleIntervalMs_;
    nvmlDevice_t device_;
    std::atomic<bool> running_;
    std::thread samplerThread_;
    std::vector<double> samples_;
    mutable std::mutex mutex_;

    double minPower_;
    double maxPower_;
};
