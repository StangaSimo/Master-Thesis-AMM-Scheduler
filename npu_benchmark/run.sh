#!/bin/bash
sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
sudo chmod a+r /sys/class/powercap/intel-rapl:0/intel-rapl:0:0/energy_uj
sudo chmod a+r /sys/class/powercap/intel-rapl:0/intel-rapl:0:1/energy_uj

make run