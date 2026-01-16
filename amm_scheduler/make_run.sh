mkdir bin 
mkdir build
rm -rf build/*
rm bin/scheduler
rm -rf bin/lib/*
cd build/ 

export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
cmake -DUSE_CUDA=ON -DUSE_SYCL=ON -DUSE_OPENVINO=OFF -DUSE_OPENBLAS=ON -DUSE_OPENCV=ON ..
make -j$(nproc)

cd ..
#python3 merge_clangd.py
./bin/scheduler
