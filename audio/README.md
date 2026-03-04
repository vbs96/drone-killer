prereq:

sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg libsndfile1

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip wheel setuptools
# Transformers + CPU torch + audio deps
pip install transformers librosa soundfile numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu

running:

source .venv/bin/activate
python3 drone_detect.py ./test.wav --out result.json