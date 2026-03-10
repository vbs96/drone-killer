# Setup

sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel
pip install "transformers>=4.41" "optimum[onnxruntime]>=1.20" onnxruntime soundfile librosa numpy

# Export the model to ONNX (do this once)
optimum-cli export onnx \
  --model preszzz/drone-audio-detection-05-17-trial-0 \
  --task audio-classification \
  onnx_drone_model

# Running:

source .venv/bin/activate
mkfifo /tmp/sim_mic_fifo

## Terminal1
python3 micsim.py --background background.wav --drone drone.wav > /tmp/sim_mic_fifo

## Terminal2
python3 drone_detect_live.py --input-fifo /tmp/sim_mic_fifo --onnx_dir onnx_drone_model