FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git wget curl unzip ffmpeg libgl1-mesa-glx \
    python3-pip python3-dev python3-venv python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

COPY ../../../github/vlm-fine-tune/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

CMD ["python", "train.py"]
