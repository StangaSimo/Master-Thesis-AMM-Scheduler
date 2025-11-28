rm build/*
rm bin/scheduler
rm bin/lib/*
cd build/ 
#cmake ..
cmake -DUSE_CUDA=ON -DUSE_SYCL=ON -DUSE_OPENVINO=ON ..
make 
cd ..
./bin/scheduler

