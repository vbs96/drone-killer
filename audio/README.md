# Setup

sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel
pip install "transformers>=4.41" "optimum[onnxruntime]>=1.20" onnxruntime soundfile librosa numpy

# Export the model to ONNX (do this once)
pip install transformers librosa soundfile numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Running:

source .venv/bin/activate
python3 drone_detect.py ./test.wav --out result.json