FROM --platform=linux/arm64 debian:bookworm AS builder

# install build tools
RUN apt-get update && \
    apt-get install -y cmake g++ git wget unzip pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev \
    libflatbuffers-dev

# build OpenCV from source
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
    unzip opencv.zip && \
    cd opencv-4.x && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui \
          -DBUILD_SHARED_LIBS=ON \
          -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
          -DBUILD_PROTOBUF=OFF \
          .. && \
    make -j$(nproc) && make install && \
    ldconfig && rm -rf /opencv-4.x /opencv.zip

# build TensorFlow Lite C library from source
RUN git clone --depth 1 --branch v2.18.0 https://github.com/tensorflow/tensorflow.git /tf && \
    cd /tf && mkdir build && cd build && \
    cmake ../tensorflow/lite/c \
          -DCMAKE_BUILD_TYPE=Release \
          -DTFLITE_ENABLE_XNNPACK=OFF && \
    make -j$(nproc) && \
    find . -name 'libtensorflowlite_c*' -type f | xargs -I{} cp {} /usr/local/lib/ && \
    ldconfig && \
    cd /tf && \
    find tensorflow/lite -name '*.h' | while read h; do \
      mkdir -p "/usr/local/include/$(dirname "$h")"; \
      cp "$h" "/usr/local/include/$h"; \
    done && \
    ldconfig && rm -rf /tf

# compile the application
WORKDIR /app
COPY CMakeLists.txt demo.cpp ./
COPY model/drone_detect.tflite model/
COPY model/anchors.bin model/
COPY model/object-detection.pbtxt model/
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DTFLITE_DIR=/usr/local && \
    cmake --build build

# ── Runtime stage ─────────────────────────────────────────────────────
FROM --platform=linux/arm64 debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libavcodec59 libavformat59 libswscale6 libflatbuffers2 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/libtensorflow* /usr/local/lib/
COPY --from=builder /usr/local/lib/libopencv*.so* /usr/local/lib/
COPY --from=builder /app/build/demo .
COPY --from=builder /app/model/ model/
RUN ldconfig

ENTRYPOINT ["./demo"]


