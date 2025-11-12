#include <opencv2/core.hpp>     /* opencv supportu */
#include <opencv2/core/ocl.hpp> /* opencl support */
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <cmath>     
#include <iomanip>  
#include <sstream>
#include <algorithm>
#include <cctype>

// Definiamo DEBUG anche qui per stampare i risultati
#define DEBUG 
#define INTEL 

void benchmark_cpu(const std::string &device_name, int M, int N, int K, int runs) {
    
#ifdef DEBUG
    std::cout << "\n=== Benchmark: " << device_name << " ===\n";
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << " | Runs: " << runs << "\n";
#endif

    cv::Mat h_A(M, K, CV_32F);
    cv::Mat h_B(K, N, CV_32F);
    cv::Mat h_C(M, N, CV_32F); 
    cv::RNG rng(42); 
    rng.fill(h_A, cv::RNG::UNIFORM, -1.0f, 1.0f);
    rng.fill(h_B, cv::RNG::UNIFORM, -1.0f, 1.0f);

    double sum_ms = 0.0;
    double sum_gflops = 0.0;

    cv::gemm(h_A, h_B, 1.0, cv::Mat(), 0.0, h_C);
    cv::ocl::finish();

    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        cv::gemm(h_A, h_B, 1.0, cv::Mat(), 0.0, h_C);
        cv::ocl::finish();

        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * M * N * K) / (ms * 1e6);

        sum_ms += ms;
        sum_gflops += gflops;
    }

    double avg_time = sum_ms / runs;
    double avg_gflops = sum_gflops / runs;

#ifdef DEBUG
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time: " << avg_time << " ms\n";
    std::cout << "Average GFLOPS: " << avg_gflops << "\n";
    std::cout << "================================\n\n";
#endif
}

void benchmark_gpu(const std::string &device_name, int M, int N, int K, int runs) {
    
    /* activate openCL */
    if (!cv::ocl::haveOpenCL()) {
        std::cout << "[ERROR] OpenCL non è disponibile o non è abilitato in OpenCV.\n";
        return;
    }
    cv::ocl::setUseOpenCL(true); 

#ifdef DEBUG
    std::cout << "\n=== Benchmark: " << device_name << " ===\n";
    std::cout << "Context: " << cv::ocl::useOpenCL() << " | Device: " << cv::ocl::Context::getDefault().device(0).name() << "\n";
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << " | Runs: " << runs << "\n";
#endif

    /* Host */
    cv::Mat h_A(M, K, CV_32F); 
    cv::Mat h_B(K, N, CV_32F);
    cv::RNG rng(42);
    rng.fill(h_A, cv::RNG::UNIFORM, -1.0f, 1.0f);
    rng.fill(h_B, cv::RNG::UNIFORM, -1.0f, 1.0f);

    /* GPU */
    cv::UMat u_A; 
    cv::UMat u_B;
    cv::UMat u_C(M, N, CV_32F); 

    /* Host to GPU */
    h_A.copyTo(u_A);
    h_B.copyTo(u_B);

    double sum_ms = 0.0;
    double sum_gflops = 0.0;

    /* warmup */
    cv::gemm(u_A, u_B, 1.0, cv::UMat(), 0.0, u_C);
    cv::ocl::finish();
    //cv::Mat warmup_result = u_C.getMat(cv::ACCESS_READ);


    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        /* umat for GPU */
        cv::gemm(u_A, u_B, 1.0, cv::UMat(), 0.0, u_C);
        cv::ocl::finish();
        auto end = std::chrono::high_resolution_clock::now();
        
        cv::Mat result_sync = u_C.getMat(cv::ACCESS_READ);
        

        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = (2.0 * M * N * K) / (ms * 1e6);

        sum_ms += ms;
        sum_gflops += gflops;
    }

    double avg_time = sum_ms / runs;
    double avg_gflops = sum_gflops / runs;

#ifdef DEBUG
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Average time: " << avg_time << " ms\n";
    std::cout << "Average GFLOPS: " << avg_gflops << "\n";
    std::cout << "================================\n\n";
#endif
}

bool createOpenCLContext(const std::string& platform_substr) {
    if (!cv::ocl::haveOpenCL()) {
        std::cerr << "[ERROR] OpenCL is not available" << std::endl;
        return false;
    }

    std::vector<cv::ocl::PlatformInfo> platforms;
    cv::ocl::getPlatfomsInfo(platforms);
   
    if (platforms.empty()) {
        std::cerr << "[ERROR] No OpenCL platforms found." << std::endl;
        return false;
    }

    std::cout << "[INFO] Available OpenCL platforms:\n";
    for (size_t i = 0; i < platforms.size(); i++) {
        std::string platformName = platforms[i].name();
        std::cout << "  [" << i << "] " << platformName;
        
        // Try to get device count for this platform
        cv::ocl::Device device;
        try {
            platforms[i].getDevice(device, 0);
            std::cout << " - Device: " << device.name();
        } catch (...) {
            std::cout << " - No devices";
        }
        std::cout << "\n";
    }

    // Find the platform
    cv::ocl::PlatformInfo foundPlatform;
    bool platformFound = false;

    for (const auto& platform : platforms) {
        std::string platformName = platform.name();
        std::string platformNameLower = platformName;
        std::transform(platformNameLower.begin(), platformNameLower.end(), 
                      platformNameLower.begin(), ::tolower);
        
        std::string substr_lower = platform_substr;
        std::transform(substr_lower.begin(), substr_lower.end(), 
                      substr_lower.begin(), ::tolower);

        if (platformNameLower.find(substr_lower) != std::string::npos) {
            foundPlatform = platform;
            platformFound = true;
            break;
        }
    }

    if (!platformFound) {
        std::cerr << "[ERROR] Could not find platform containing: " << platform_substr << std::endl;
        return false;
    }

    std::cout << "[INFO] Selected platform: " << foundPlatform.name() << "\n";

    // Get the device from this platform
    cv::ocl::Device device;
    foundPlatform.getDevice(device, 0);
    
    std::cout << "[INFO] Selected device: " << device.name() << "\n";
    std::cout << "[INFO] Device vendor: " << device.vendorName() << "\n";

    // Create context using the static method with configuration string
    // Format: "platform_name:device_type:device_index"
    std::stringstream config;
    config << foundPlatform.name() << ":GPU:0";
    
    cv::ocl::Context context = cv::ocl::Context::create(config.str());
    
    if (context.empty()) {
        std::cerr << "[ERROR] Failed to create context with config: " << config.str() << std::endl;
        return false;
    }

    std::cout << "[INFO] Context created successfully\n";
    
    cv::ocl::setUseOpenCL(true);

    cv::ocl::Device currentDevice = cv::ocl::Device::getDefault();
    std::cout << "[INFO] OpenCV is now using: " << currentDevice.name() << "\n";
    std::cout << "[INFO] OpenCL enabled: " << (cv::ocl::useOpenCL() ? "YES" : "NO") << "\n";

    return true;
}

int main() {
    int M = 1024; 
    int N = 1024; 
    int K = 512;
    int runs = 5;

#ifdef DEBUG 
    /* Lapack */
    std::cout << cv::getBuildInformation() << std::endl;
#endif  

#ifdef INTEL
    if (!createOpenCLContext("intel")) {
        std::cerr << "[WARNING] Failed to set Intel GPU context, using default device\n";
    }
#endif

    benchmark_gpu("GPU (OpenCV/OpenCL)", M, N, K, runs);
    benchmark_cpu("CPU (OpenCV)", M, N, K, runs);
    benchmark_gpu("GPU (OpenCV/OpenCL)", 2048, 2048, 1024, runs);
    benchmark_cpu("CPU (OpenCV)", 2048, 2048, 1024, runs);
    //TODO: aggiungere TLB, MKL e non so la roba di amd che veramente non usare e non potrò manco far funzionare, comunque sia aggiungere a tutto questo anche il fatto del consumo di watt e tutte le statistiche maronne 
    return 0;
}