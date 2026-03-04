cmake_minimum_required(VERSION 3.14)
project(drone_detection_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find OpenCV ──────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── Find TensorFlow C library ────────────────────────────────────────
# Set TENSORFLOW_DIR to the root of the extracted libtensorflow package,
# or install it system-wide so that the headers/libs are on default paths.
#
#   cmake -DTENSORFLOW_DIR=/path/to/libtensorflow ..
#
if(DEFINED TENSORFLOW_DIR)
    set(TF_INCLUDE_DIR "${TENSORFLOW_DIR}/include")
    set(TF_LIB_DIR     "${TENSORFLOW_DIR}/lib")
else()
    # Try pkg-config as a fallback
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TF QUIET tensorflow)
    endif()
    if(NOT TF_FOUND)
        # Assume system-installed
        set(TF_INCLUDE_DIR "")
        set(TF_LIB_DIR     "")
    endif()
endif()

# ── Build ─────────────────────────────────────────────────────────────
add_executable(demo demo.cpp)

target_include_directories(demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TF_INCLUDE_DIR}
)

target_link_directories(demo PRIVATE
    ${TF_LIB_DIR}
)

target_link_libraries(demo PRIVATE
    ${OpenCV_LIBS}
    tensorflow
)
