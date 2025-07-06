@echo off
REM Drug Discovery Compound Optimization - Pip Installation Script (Windows)
REM This script installs all dependencies using pip only

echo 🧬 Installing Drug Discovery Compound Optimization Dependencies...

REM Check if Python is installed
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python is not installed or not in PATH.
    echo    Please install Python 3.10+ from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version

echo 📦 Upgrading pip...
python -m pip install --upgrade pip

echo 🔥 Installing PyTorch (CPU version)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo 📋 Installing requirements from requirements.txt...
pip install -r requirements.txt

echo 🧪 Testing installation...
python -c "import torch; print('✅ PyTorch installed successfully')"
python -c "import pandas; print('✅ Pandas installed successfully')"
python -c "import numpy; print('✅ NumPy installed successfully')"
python -c "import sklearn; print('✅ Scikit-learn installed successfully')"

REM Test optional packages
python -c "import rdkit; print('✅ RDKit installed successfully')" 2>nul || echo "⚠️  RDKit not available - some functionality will be limited"
python -c "import deepchem; print('✅ DeepChem installed successfully')" 2>nul || echo "⚠️  DeepChem not available - some functionality will be limited"

echo.
echo ✅ Installation complete!
echo.
echo 🚀 To test the installation, run:
echo    python src/data_processing.py
echo    python src/models.py
echo    python src/utils.py
echo.
echo 🌐 To start the API server, run:
echo    python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
echo.
echo 📊 To start Jupyter Lab, run:
echo    jupyter lab
echo.
echo 🧬 Happy drug discovery! 🧬
pause