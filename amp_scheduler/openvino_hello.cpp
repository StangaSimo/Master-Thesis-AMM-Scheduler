#include "openvino_hello.h"
#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>

void runOpenvinoHello() {
    ov::Core core;
    std::string device_name = "GPU";
    bool available = false;

    for (auto &d : core.get_available_devices())
        if (d.find(device_name) != std::string::npos)
            available = true;
    if (!available) {
        std::cout << "Device " << device_name << " non disponibile\n";
        return;
    }
}