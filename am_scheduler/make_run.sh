rm -rf build/*
rm bin/scheduler
rm -rf bin/lib/*
cd build/ 

export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
cmake -DUSE_CUDA=ON -DUSE_SYCL=ON -DUSE_OPENVINO=ON ..
make -j$(nproc)

#cmake -G Ninja -DUSE_CUDA=ON -DUSE_SYCL=ON -DUSE_OPENVINO=ON ..
#ninja

cd ..
#python3 merge_clangd.py
./bin/scheduler
