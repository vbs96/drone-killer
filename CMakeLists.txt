cmake_minimum_required(VERSION 3.14)
project(drone_detection_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find OpenCV ──────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── Find TensorFlow Lite C library ───────────────────────────────────
# Set TFLITE_DIR to the root of the TFLite install prefix,
# or install it system-wide so that the headers/libs are on default paths.
#
#   cmake -DTFLITE_DIR=/path/to/tflite ..
#
if(DEFINED TFLITE_DIR)
    set(TFLITE_INCLUDE_DIR "${TFLITE_DIR}/include")
    set(TFLITE_LIB_DIR     "${TFLITE_DIR}/lib")
else()
    set(TFLITE_INCLUDE_DIR "")
    set(TFLITE_LIB_DIR     "")
endif()

# ── Build ─────────────────────────────────────────────────────────────
add_executable(demo demo.cpp)

target_include_directories(demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TFLITE_INCLUDE_DIR}
)

target_link_directories(demo PRIVATE
    ${TFLITE_LIB_DIR}
)

target_link_libraries(demo PRIVATE
    ${OpenCV_LIBS}
    tensorflowlite_c
)
