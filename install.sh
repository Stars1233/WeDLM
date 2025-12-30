#!/bin/bash
set -e

# WeDLM Installation Script
# Installs dependencies in correct order (torch -> flash-attn -> wedlm)

CUDA_VERSION="${CUDA_VERSION:-cu129}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"

echo "=== WeDLM Installation ==="
echo "CUDA: $CUDA_VERSION | PyTorch: $TORCH_VERSION | flash-attn: $FLASH_ATTN_VERSION"
echo ""

# Step 1: PyTorch
echo "[1/4] Installing PyTorch..."
pip install torch==${TORCH_VERSION}+${CUDA_VERSION} --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Step 2: flash-attn build dependencies
echo "[2/4] Installing flash-attn build dependencies..."
pip install psutil ninja packaging

# Step 3: flash-attn (requires torch, needs compilation)
echo "[3/4] Installing flash-attn..."
pip install flash-attn==${FLASH_ATTN_VERSION} --no-build-isolation

# Step 4: WeDLM
echo "[4/4] Installing WeDLM..."
pip install -e .

echo ""
echo "=== Done ==="

