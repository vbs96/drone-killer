# drone-killer

Drone detection system using an SSD MobileNet v1 model converted to TensorFlow Lite. Processes video files or live camera feeds, draws bounding boxes around detected drones, and optionally saves the annotated output.

## Usage

```
./demo <input> [output_path]
```

| Input type | Argument example | Description |
|---|---|---|
| Video file | `./demo video.mp4 output.mp4` | Process a video file and save annotated output |
| Camera index | `./demo 0` | Open `/dev/video0` (V4L2 / USB camera) |
| Camera + record | `./demo 0 output.mp4` | Live camera with recording to file |
| GStreamer pipeline | `./demo "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink"` | Pi libcamera stack (Bullseye+) |

- **Video file mode:** processes all frames, writes output, and exits.
- **Live camera mode:** displays a window with detections in real-time. Press `q` to quit. Output file is optional.

## Build (native, on ARM64 Debian 12 or 13 VM)

### Prerequisites

```bash
sudo apt update
sudo apt install -y cmake g++ git wget unzip pkg-config \
    libopencv-dev \
    libavcodec-dev libavformat-dev libswscale-dev
```

### Build TFLite C library (same as image-comp-installer.sh)

```bash
git clone --depth 1 --branch v2.18.0 https://github.com/tensorflow/tensorflow.git /tmp/tf
cd /tmp/tf && mkdir build && cd build
cmake ../tensorflow/lite/c \
    -DCMAKE_BUILD_TYPE=Release \
    -DTFLITE_ENABLE_XNNPACK=OFF
make -j$(nproc)

sudo find . -name 'libtensorflowlite_c*' -type f -exec cp {} /usr/local/lib/ \;
cd /tmp/tf
sudo find tensorflow/lite -name '*.h' -exec sh -c \
    'mkdir -p "/usr/local/include/$(dirname "$1")" && cp "$1" "/usr/local/include/$1"' _ {} \;
sudo ldconfig
rm -rf /tmp/tf
```

### Build the project

```bash
cd ~/drone-killer
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTFLITE_DIR=/usr/local
cmake --build build
```

### Test

```bash
./build/demo V_DRONE_024.mp4 output.mp4
```

## Deploy to Raspberry Pi

Copy the following to the Pi:

```
build/demo                        # binary
model/drone_detect.tflite         # TFLite model (no Flex ops)
model/anchors.bin                 # precomputed anchor boxes
model/object-detection.pbtxt      # label map
```

On the Pi, install runtime dependencies:

```bash
sudo apt install libopencv-dev libavcodec-dev libavformat-dev libswscale-dev
```

Run:

```bash
# From a video file
./demo video.mp4 output.mp4

# From USB camera
./demo 0

# From Pi camera (libcamera)
./demo "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink"
```

## Model

SSD MobileNet v1 trained for drone detection. The TFLite model outputs raw box encodings and class logits — post-processing (box decoding, sigmoid, NMS) is done in C++.

- Input: 300x300x3 float32, normalized to [-1, 1]
- Output 0: raw box encodings [1, 1917, 4]
- Output 1: raw class scores [1, 1917, 2] (background + drone)