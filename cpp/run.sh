# force delete the build directory
# rm -rf build
# create the build directory
# mkdir build
# change to the build directory
cd build
# run cmake
cmake ..
# run make
make -j4
cd ../../cpp_py
python3 main.py
# python3 train.py