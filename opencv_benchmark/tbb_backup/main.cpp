#define DEBUG 
//#define DEBUG_MACHINE
#define INTEL 

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
#ifdef INTEL
#include <mkl.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#endif

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

void benchmark_mkl_cpu(const std::string &device_name, int M, int N, int K, int runs) {
    
#ifdef DEBUG
    std::cout << "\n=== Benchmark: " << device_name << " (CPU + oneMKL) ===\n";
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << " | Runs: " << runs << "\n";
#endif

    // 1. Allocazione Matrici Host (cv::Mat è comodo)
    // oneMKL richiede un layout Row-Major o Col-Major.
    // cv::Mat è Row-Major, quindi useremo CblasRowMajor.
    cv::Mat h_A(M, K, CV_32F);
    cv::Mat h_B(K, N, CV_32F);
    cv::Mat h_C(M, N, CV_32F); // Matrice risultato
    
    cv::RNG rng(42);
    rng.fill(h_A, cv::RNG::UNIFORM, -1.0f, 1.0f);
    rng.fill(h_B, cv::RNG::UNIFORM, -1.0f, 1.0f);

    // Puntatori ai dati (necessari per l'API C di MKL)
    const float* pA = (const float*)h_A.data;
    const float* pB = (const float*)h_B.data;
    float* pC = (float*)h_C.data;

    // 2. Parametri per sgemm (Single-precision General Matrix Multiply)
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Le "Leading Dimensions" (lda, ldb, ldc) sono FONDAMENTALI.
    // Per CblasRowMajor:
    // lda = numero di colonne di A = K
    // ldb = numero di colonne di B = N
    // ldc = numero di colonne di C = N
    MKL_INT lda = K;
    MKL_INT ldb = N;
    MKL_INT ldc = N;

    // 3. Warmup
    cblas_sgemm(
        CblasRowMajor,  // Layout
        CblasNoTrans,   // No Trasposta A
        CblasNoTrans,   // No Trasposta B
        M, N, K,        // Dimensioni
        alpha,          // alpha
        pA, lda,        // Matrice A
        pB, ldb,        // Matrice B
        beta,           // beta
        pC, ldc         // Matrice C
    );

    // 4. Benchmark
    double sum_ms = 0.0;
    double sum_gflops = 0.0;

    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            alpha,
            pA, lda,
            pB, ldb,
            beta,
            pC, ldc
        );

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

void benchmark_mkl_cpu_tbb(const std::string &device_name, int M, int N, int K, int runs, int num_threads) {
    
    std::unique_ptr<oneapi::tbb::global_control> tbb_limiter;
    
    // Se num_threads > 0, impostiamo un limite. 
    // Altrimenti, TBB userà il suo default (di solito tutt  // Controlla quanti thread massimi TBB può usare in totale
    // Usiamo std::unique_ptr per gestire correttamente la vita dell'oggettoi i core)
    if (num_threads > 0) {
        tbb_limiter = std::make_unique<oneapi::tbb::global_control>(
            oneapi::tbb::global_control::max_allowed_parallelism, num_threads
        );
    }
    
    int actual_threads = oneapi::tbb::global_control::active_value(
                             oneapi::tbb::global_control::max_allowed_parallelism
                         );

#ifdef DEBUG
    std::cout << "\n=== Benchmark: " << device_name << " (CPU + oneMKL-TBB) ===\n";
    std::cout << "Target Threads: " << (num_threads > 0 ? std::to_string(num_threads) : "Default") 
              << " | Active Threads: " << actual_threads << "\n";
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << " | Runs: " << runs << "\n";
#endif

    // 1. Allocazione Matrici (identica a prima)
    cv::Mat h_A(M, K, CV_32F);
    cv::Mat h_B(K, N, CV_32F);
    cv::Mat h_C(M, N, CV_32F); 
    
    cv::RNG rng(42);  // Controlla quanti thread massimi TBB può usare in totale
    // Usiamo std::unique_ptr per gestire correttamente la vita dell'oggetto
    rng.fill(h_A, cv::RNG::UNIFORM, -1.0f, 1.0f);
    rng.fill(h_B, cv::RNG::UNIFORM, -1.0f, 1.0f);

    const float* pA = (const float*)h_A.data;
    const float* pB = (const float*)h_B.data;
    float* pC = (float*)h_C.data;

    // 2. Parametri per sgemm (identici a prima)
    float alpha = 1.0f;
    float beta = 0.0f;
    MKL_INT lda = K;
    MKL_INT ldb = N;
    MKL_INT ldc = N;

    // 3. Creiamo un'arena TBB per eseguire il lavoro
    // Questo assicura che il nostro sgemm giri nel contesto TBB
    // che abbiamo appena configurato.
    oneapi::tbb::task_arena arena(num_threads > 0 ? num_threads : tbb::task_arena::automatic);

    // 4. Warmup
    arena.execute([&]() {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, alpha, pA, lda, pB, ldb, beta, pC, ldc
        );
    });


    // 5. Benchmark
    double sum_ms = 0.0;
    double sum_gflops = 0.0;

    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        
        // Eseguiamo il gemm *dentro* l'arena TBB
        arena.execute([&]() {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, pA, lda, pB, ldb, beta, pC, ldc
            );
        });

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

int main() {
    int M = 4096; 
    int N = 4096; 
    int K = 1024;
    int runs = 1;

#ifdef DEBUG_MACHINE
    /* Lapack */
    std::cout << cv::getBuildInformation() << std::endl;
#endif  

#ifdef INTEL
    if (!createOpenCLContext("intel")) {
        std::cerr << "[WARNING] Failed to set Intel GPU context, using default device\n";
    }
#endif

    benchmark_mkl_cpu("CPU (oneMKL)", M, N, K, runs);
    benchmark_gpu("GPU (OpenCV/OpenCL)", M, N, K, runs);
    benchmark_cpu("CPU (OpenCV)", M, N, K, runs);

    benchmark_mkl_cpu_tbb("CPU (MKL 1-Thread)", M, N, K, runs, 1);
    benchmark_mkl_cpu_tbb("CPU (MKL 4-Threads)", M, N, K, 30, 4);
    benchmark_mkl_cpu_tbb("CPU (MKL Max-Threads)", M, N, K, runs, 16);
    //TODO: aggiungere TLB, MKL e non so la roba di amd che veramente non usare e non potrò manco far funzionare, comunque sia aggiungere a tutto questo anche il fatto del consumo di watt e tutte le statistiche maronne 
    return 0;
}