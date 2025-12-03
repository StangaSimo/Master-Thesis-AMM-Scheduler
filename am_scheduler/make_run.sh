rm -rf build/*
rm bin/scheduler
rm -rf bin/lib/*
cd build/ 
#cmake ..
#cmake -DUSE_CUDA=ON -DUSE_SYCL=ON -DUSE_OPENVINO=ON ..
cmake -DUSE_CUDA=ON -DUSE_SYCL=ON -DUSE_OPENVINO=ON ..
make 
cd ..
#python3 merge_clangd.py
./bin/scheduler

