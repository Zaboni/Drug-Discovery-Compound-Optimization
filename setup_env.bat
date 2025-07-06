@echo off
REM Drug Discovery Compound Optimization Environment Setup Script (Windows)
REM This script creates a conda environment with all necessary dependencies

echo 🧬 Setting up Drug Discovery Compound Optimization Environment...

REM Environment name
set ENV_NAME=drug-discovery

REM Check if conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda is not installed. Please install Anaconda or Miniconda first.
    echo    Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Remove existing environment if it exists
conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% equ 0 (
    echo 🗑️  Removing existing %ENV_NAME% environment...
    conda env remove -n %ENV_NAME% -y
)

echo 🔧 Creating new conda environment: %ENV_NAME%
conda create -n %ENV_NAME% python=3.10 -y

echo 🔄 Activating environment...
call conda activate %ENV_NAME%

echo 📦 Installing conda packages...
REM Install conda packages first (these are often more stable from conda-forge)
conda install -c conda-forge -y numpy pandas matplotlib scikit-learn jupyter jupyterlab pyyaml requests tqdm

echo 🔥 Installing PyTorch (CPU version)...
REM Install PyTorch CPU version
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

echo 🧪 Installing chemistry packages...
REM Install RDKit from conda-forge (recommended way)
conda install -c conda-forge rdkit -y

echo 📋 Installing remaining packages with pip...
REM Install remaining packages with pip
pip install --upgrade pip
pip install -r requirements.txt

echo ✅ Environment setup complete!
echo.
echo 🚀 To activate the environment, run:
echo    conda activate %ENV_NAME%
echo.
echo 🔬 To deactivate the environment, run:
echo    conda deactivate
echo.
echo 📝 Optional GPU Setup Instructions:
echo    If you have a CUDA-compatible GPU and want to use GPU acceleration:
echo    1. First, check your CUDA version: nvidia-smi
echo    2. Uninstall CPU PyTorch: pip uninstall torch torchvision torchaudio
echo    3. Install GPU PyTorch (replace cu118 with your CUDA version):
echo       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo    4. Install additional GPU packages:
echo       pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
echo.
echo 🧬 Happy drug discovery! 🧬
pause