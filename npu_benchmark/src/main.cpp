#include <openvino/openvino.hpp>
#include <iostream>

int main() {
    try {
        ov::Core core;
        auto devices = core.get_available_devices();

        std::cout << "Dispositivi disponibili:" << std::endl;
        for (const auto& device : devices) {
            std::cout << " - " << device << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "Errore: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}