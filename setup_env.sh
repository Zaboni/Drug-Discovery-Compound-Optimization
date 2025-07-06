#!/bin/bash

# Drug Discovery Compound Optimization Environment Setup Script
# This script creates a conda environment with all necessary dependencies

set -e  # Exit on any error

echo "🧬 Setting up Drug Discovery Compound Optimization Environment..."

# Environment name
ENV_NAME="drug-discovery"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "🗑️  Removing existing ${ENV_NAME} environment..."
    conda env remove -n ${ENV_NAME} -y
fi

echo "🔧 Creating new conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.10 -y

echo "🔄 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "📦 Installing conda packages..."
# Install conda packages first (these are often more stable from conda-forge)
conda install -c conda-forge -y \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    jupyter \
    jupyterlab \
    pyyaml \
    requests \
    tqdm

echo "🔥 Installing PyTorch (CPU version)..."
# Install PyTorch CPU version
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

echo "🧪 Installing chemistry packages..."
# Install RDKit from conda-forge (recommended way)
conda install -c conda-forge rdkit -y

echo "📋 Installing remaining packages with pip..."
# Install remaining packages with pip
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete!"
echo ""
echo "🚀 To activate the environment, run:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo "🔬 To deactivate the environment, run:"
echo "   conda deactivate"
echo ""
echo "📝 Optional GPU Setup Instructions:"
echo "   If you have a CUDA-compatible GPU and want to use GPU acceleration:"
echo "   1. First, check your CUDA version: nvidia-smi"
echo "   2. Uninstall CPU PyTorch: pip uninstall torch torchvision torchaudio"
echo "   3. Install GPU PyTorch (replace cu118 with your CUDA version):"
echo "      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo "   4. Install additional GPU packages:"
echo "      pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html"
echo ""
echo "🧬 Happy drug discovery! 🧬"