#ifndef OPENCV_TEST_H
#define OPENCV_TEST_H


#include <cstdlib>
#include <vector>
#ifdef ENABLE_OPENCV 
#include <opencv2/opencv.hpp>
#endif

#include "tasks.hpp"
#include "scheduler.hpp"

inline void prepare_task_from_frame(const cv::Mat& frame, const cv::Mat& kernel, task& t) {
    int k_sz = kernel.rows; // Assumiamo kernel quadrato (es. 3)

    // 1. Calcolo Dimensioni Output (N)
    int out_h = frame.rows - k_sz + 1;
    int out_w = frame.cols - k_sz + 1;
    int num_patches = out_h * out_w; // Questo è il tuo N

    // 2. Setup Metadati Task
    t.type = Type::FLOAT;
    t.M = 1;                    // Filtro srotolato è 1 riga
    t.K = k_sz * k_sz;          // Lunghezza filtro (es. 9)
    t.N = num_patches;          // Colonne di B (totale pixel output)

    // 3. Allocazione Memoria (Se non è già allocata)
    // NOTA: In produzione, evita di fare malloc/new a ogni frame! Riusa i buffer.
    // A: 1 * K
    t.A = new float[t.M * t.K];
    // B: K * N
    t.B = new float[t.K * t.N]; 
    // C: 1 * N (M * N)
    t.C = new float[t.M * t.N];

    // 4. Riempimento Matrice A (Il Filtro)
    // Copiamo il kernel appiattendolo
    float* A_ptr = (float*)t.A;
    int a_idx = 0;
    for(int i=0; i<k_sz; ++i) {
        for(int j=0; j<k_sz; ++j) {
            A_ptr[a_idx++] = kernel.at<float>(i, j);
        }
    }

    // 5. Riempimento Matrice B (im2col - Image to Column)
    // Vogliamo che B sia (K righe x N colonne).
    // Ogni colonna j rappresenta un patch dell'immagine.
    // Ogni riga i rappresenta un pixel specifico del kernel (es. l'angolo in alto a sinistra).

    float* B_ptr = (float*)t.B;
    int patch_idx = 0; // Indice della colonna corrente (da 0 a N-1)

    // Scorriamo l'immagine (sliding window)
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {

            // Per ogni patch, copiamo i pixel nelle righe corrispondenti della colonna patch_idx
            int k_pixel_idx = 0; // Indice della riga in B (da 0 a K-1)

            for (int ky = 0; ky < k_sz; ky++) {
                for (int kx = 0; kx < k_sz; kx++) {
                    float pixel_val = frame.at<float>(y + ky, x + kx);

                    // FORMULA MATRICE FLATTENED: index = (riga * larghezza) + colonna
                    // riga = k_pixel_idx
                    // larghezza = t.N (numero di colonne totali)
                    // colonna = patch_idx
                    B_ptr[k_pixel_idx * t.N + patch_idx] = pixel_val;

                    k_pixel_idx++;
                }
            }
            patch_idx++;
        }
    }
}

inline cv::Mat result_from_task(const task& t, int original_w, int kernel_sz) {
    int out_h = (t.N / (original_w - kernel_sz + 1));
    int out_w = (original_w - kernel_sz + 1);
    cv::Mat result(out_h, out_w, CV_32F, t.C);
    return result.clone(); 
}

inline void free_batch(std::vector<task>& tasks) {
    for(auto& t : tasks) {
        delete[] (float*)t.A;
        delete[] (float*)t.B;
        delete[] (float*)t.C;
    }
    tasks.clear();
}

inline void test_video_filter() {
    cv::VideoCapture cap("test.mp4");

    if(!cap.isOpened()) {
        cout << "[OPENCV TEST] ERROR video no present\n";
        exit(EXIT_FAILURE);
    }

    /* edge detection */
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);

    cv::Mat frame, gray, float_img;
    int i=0; 

    bool video_ended = false;

    vector<task> tasks;
    tasks.reserve(1000);
    AMScheduler scheduler = AMScheduler(Logic::STATIC_PARTITIONING);

    while(true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        } 

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(float_img, CV_32F);
        task t;
        prepare_task_from_frame(float_img, kernel, t);

        tasks.push_back(t);
    }

    cout << "[OPENCV TEST] frame Processati\n";
    scheduler.do_tasks(tasks.data(), tasks.size());
    scheduler.wait();
    cout << "[OPENCV TEST] Scheduler finito\n";

    for (auto& t : tasks) {
        cv::Mat display;

        cv::Mat out = result_from_task(t, float_img.cols, kernel.rows);

        cv::normalize(out, out, 0, 255, cv::NORM_MINMAX); 
        out.convertTo(display, CV_8U);

        cv::imshow("Scheduler Output", display);

        if (cv::waitKey(30) == 27) {
            video_ended = true;
            break;
        }
    }

    free_batch(tasks);

    cout << "[OPENCV TEST] frame finiti\n";
}

#endif
