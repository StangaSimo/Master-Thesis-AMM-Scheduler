#include <CL/sycl.hpp>
#include <iostream>

int main() {
    try {
        // Creiamo un "selector" per dispositivi Level-Zero
        sycl::queue q(sycl::gpu_selector{}); // prova gpu_selector o default_selector
        auto dev = q.get_device();

        std::cout << "Usando dispositivo: " 
                  << dev.get_info<sycl::info::device::name>() << "\n";

        const int N = 16;
        std::vector<int> data(N, 1);

        {
            // Buffer SYCL
            sycl::buffer<int> buf(data.data(), sycl::range<1>(N));

            q.submit([&](sycl::handler &h) {
                auto acc = buf.get_access<sycl::access::mode::read_write>(h);

                h.parallel_for<class add_one>(sycl::range<1>(N), [=](sycl::id<1> i) {
                    acc[i] += 1;  // semplicissimo kernel
                });
            });
        } // fine scope buffer -> dati copiati indietro automaticamente

        std::cout << "Dati risultanti: ";
        for(auto v : data) std::cout << v << " ";
        std::cout << "\n";
    } catch(sycl::exception const &e) {
        std::cout << "Errore SYCL: " << e.what() << "\n";
    }
    return 0;
}
