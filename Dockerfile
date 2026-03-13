FROM --platform=linux/arm64 debian:bookworm AS builder

# install and update tools
RUN apt-get update && \
    apt-get install -y build-essential cmake git

# install OpenCV
RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv &&  mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && make -j8 && make install
RUN ldconfig && rm -rf opencv

# install tensorflow API
RUN FILENAME=libtensorflow-cpu-linux-x86_64.tar.gz && wget -q --no-check-certificate https://storage.googleapis.com/tensorflow/versions/2.18.1/${FILENAME} && sudo tar -C /usr/local -xzf ${FILENAME}
RUN sudo ldconfig /usr/local/lib

# compile
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DTENSORFLOW_DIR=/path/to/libtensorflow
RUN cmake --build build

FROM debian:bookworm-slim

COPY --from=builder /build/demo .
CMD ["./demo"]


