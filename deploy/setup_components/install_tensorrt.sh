#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Error: \"$BASH_COMMAND\" failed at line $LINENO" >&2' ERR

echo "⏳ Starting installation of TensorRT..."

CUDA_DETECTED="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+')" || true

if [ -z "${CUDA_DETECTED:-}" ]; then
  echo "❌ Could not detect CUDA version from nvidia-smi" >&2
  exit 1
fi

echo "✅ Detected CUDA version: $CUDA_DETECTED"

case "$CUDA_DETECTED" in
  11.*)
    echo "❌ CUDA 11.x is not supported." >&2
    exit 1
    ;;
  12.*)
    CUDA_VERSION="12.9"
    ;;
  13.*)
    CUDA_VERSION="$CUDA_DETECTED"
    ;;
  *)
    echo "❌ Unsupported CUDA version \"$CUDA_DETECTED\" (expected 12.x or 13.x)." >&2
    exit 1
    ;;
esac

echo "✅ Using CUDA_VERSION=${CUDA_VERSION}"

export TRT_VERSION=$(
  apt-cache madison tensorrt \
    | awk '{print $3}' \
    | grep -F "+cuda$CUDA_VERSION" \
    | sort -V \
    | tail -1
)

if [ -z "${TRT_VERSION:-}" ]; then
  echo "❌ Could not set TensorRT version"
  exit 1
fi

echo "✅ Using TensorRT version: $TRT_VERSION"

sudo apt-mark unhold \
libnvinfer-bin \
libnvinfer-dev \
libnvinfer-dispatch-dev \
libnvinfer-dispatch10 \
libnvinfer-headers-dev \
libnvinfer-headers-plugin-dev \
libnvinfer-headers-python-plugin-dev \
libnvinfer-lean-dev \
libnvinfer-lean10 \
libnvinfer-plugin-dev \
libnvinfer-plugin10 \
libnvinfer-vc-plugin-dev \
libnvinfer-vc-plugin10 \
libnvinfer-win-builder-resource10 \
libnvinfer10 \
libnvonnxparsers-dev \
libnvonnxparsers10 \
python3-libnvinfer-dev \
python3-libnvinfer-dispatch \
python3-libnvinfer-lean \
python3-libnvinfer \
tensorrt-dev \
tensorrt-libs \
tensorrt

sudo apt-get install -y \
libnvinfer-bin=${TRT_VERSION} \
libnvinfer-dev=${TRT_VERSION} \
libnvinfer-dispatch-dev=${TRT_VERSION} \
libnvinfer-dispatch10=${TRT_VERSION} \
libnvinfer-headers-dev=${TRT_VERSION} \
libnvinfer-headers-plugin-dev=${TRT_VERSION} \
libnvinfer-headers-python-plugin-dev=${TRT_VERSION} \
libnvinfer-lean-dev=${TRT_VERSION} \
libnvinfer-lean10=${TRT_VERSION} \
libnvinfer-plugin-dev=${TRT_VERSION} \
libnvinfer-plugin10=${TRT_VERSION} \
libnvinfer-vc-plugin-dev=${TRT_VERSION} \
libnvinfer-vc-plugin10=${TRT_VERSION} \
libnvinfer-win-builder-resource10=${TRT_VERSION} \
libnvinfer10=${TRT_VERSION} \
libnvonnxparsers-dev=${TRT_VERSION} \
libnvonnxparsers10=${TRT_VERSION} \
python3-libnvinfer-dev=${TRT_VERSION} \
python3-libnvinfer-dispatch=${TRT_VERSION} \
python3-libnvinfer-lean=${TRT_VERSION} \
python3-libnvinfer=${TRT_VERSION} \
tensorrt-dev=${TRT_VERSION} \
tensorrt-libs=${TRT_VERSION} \
tensorrt=${TRT_VERSION}

sudo apt-mark hold \
libnvinfer-bin \
libnvinfer-dev \
libnvinfer-dispatch-dev \
libnvinfer-dispatch10 \
libnvinfer-headers-dev \
libnvinfer-headers-plugin-dev \
libnvinfer-headers-python-plugin-dev \
libnvinfer-lean-dev \
libnvinfer-lean10 \
libnvinfer-plugin-dev \
libnvinfer-plugin10 \
libnvinfer-vc-plugin-dev \
libnvinfer-vc-plugin10 \
libnvinfer-win-builder-resource10 \
libnvinfer10 \
libnvonnxparsers-dev \
libnvonnxparsers10 \
python3-libnvinfer-dev \
python3-libnvinfer-dispatch \
python3-libnvinfer-lean \
python3-libnvinfer \
tensorrt-dev \
tensorrt-libs \
tensorrt

echo "✅ Installed TensorRT"
