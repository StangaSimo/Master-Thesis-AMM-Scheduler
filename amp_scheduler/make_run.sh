rm build/*
rm bin/scheduler
rm bin/lib/*
cd build/ 
cmake ..
make 
cd ..
./bin/scheduler

