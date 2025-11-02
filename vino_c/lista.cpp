#include <CL/sycl.hpp>
#include <iostream>

int main() {
    try {
        auto platforms = sycl::platform::get_platforms();
        std::cout << "Trovate " << platforms.size() << " piattaforme.\n";

        bool npu_found = false;

        for (const auto &plat : platforms) {
            std::cout << "\nPiattaforma: " << plat.get_info<sycl::info::platform::name>() << "\n";
            std::cout << "Vendor: " << plat.get_info<sycl::info::platform::vendor>() << "\n";
            std::cout << "Versione: " << plat.get_info<sycl::info::platform::version>() << "\n";

            auto devices = plat.get_devices();
            std::cout << "  Dispositivi: " << devices.size() << "\n";

            for (const auto &dev : devices) {
                std::string type;
                switch(dev.get_info<sycl::info::device::device_type>()) {
                    case sycl::info::device_type::cpu: type = "CPU"; break;
                    case sycl::info::device_type::gpu: type = "GPU"; break;
                    case sycl::info::device_type::accelerator: type = "Accelerator"; break;
                    case sycl::info::device_type::custom: type = "Custom"; break;
                    default: type = "Altro"; break;
                }

                std::cout << "    Nome: " << dev.get_info<sycl::info::device::name>() << "\n";
                std::cout << "    Tipo: " << type << "\n";
                std::cout << "    Memoria globale (MB): "
                          << dev.get_info<sycl::info::device::global_mem_size>() / (1024*1024) << "\n";
                std::cout << "    Max compute units: "
                          << dev.get_info<sycl::info::device::max_compute_units>() << "\n";

                // Controllo se è una NPU
                if (type == "Accelerator" || type == "Custom") {
                    std::cout << "    --> Possibile NPU trovata!\n";
                    npu_found = true;
                }
            }
        }

        if (!npu_found) {
            std::cout << "\nAttenzione: nessuna NPU Intel trovata sul sistema.\n";
            std::cout << "Assicurati che il driver Level-Zero e i firmware NPU siano installati e caricati.\n";
        }

    } catch (sycl::exception const &e) {
        std::cerr << "Errore SYCL: " << e.what() << "\n";
    }

    return 0;
}
