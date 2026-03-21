# ── 1. Install dependencies ──────────────────────────────────────────
sudo apt update
sudo apt install -y cmake g++ git wget unzip pkg-config \
    libopencv-dev \
    libavcodec-dev libavformat-dev libswscale-dev

# ── 2. Build TFLite C library from source ─────────────────────────────
git clone --depth 1 --branch v2.18.0 https://github.com/tensorflow/tensorflow.git /tmp/tf
cd /tmp/tf && mkdir build && cd build
cmake ../tensorflow/lite/c \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_ENABLE_XNNPACK=OFF
make -j$(nproc)

# Install the library and headers
sudo find . -name 'libtensorflowlite_c*' -type f -exec cp {} /usr/local/lib/ \;
cd /tmp/tf
sudo find tensorflow/lite -name '*.h' -exec sh -c \
    'mkdir -p "/usr/local/include/$(dirname "$1")" && cp "$1" "/usr/local/include/$1"' _ {} \;
sudo ldconfig
rm -rf /tmp/tf

# ── 3. Build the project ─────────────────────────────────────────────
cd ~/drone-killer   # wherever you cloned the repo
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTFLITE_DIR=/usr/local
cmake --build build

# ── 4. Test it ────────────────────────────────────────────────────────
./build/demo V_DRONE_024.mp4 output.mp4