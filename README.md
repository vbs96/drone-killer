# drone-killer

## Build C++

### Prereq
- OpenCV
- Tensorflow

### Cmds
- mkdir build && cd build
- cmake -DTENSORFLOW_DIR=/path/to/libtensorflow ..
- make
- cd ..
- ./build/demo