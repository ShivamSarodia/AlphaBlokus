FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN git clone https://github.com/ShivamSarodia/AlphaBlokus.git /workspace/AlphaBlokus
WORKDIR /workspace/AlphaBlokus

# Install dependencies
RUN pip install .