# 🧬 Drug Discovery Compound Optimization System

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive machine learning system for drug discovery and compound optimization using graph neural networks, molecular property prediction, and optimization algorithms.

## 🎯 Project Overview

This system provides a complete pipeline for:
- **Molecular Property Prediction**: Predict ADMET properties, bioactivity, and toxicity
- **Compound Optimization**: Generate and optimize molecular structures
- **Drug Discovery**: Identify promising drug candidates
- **Chemical Space Exploration**: Navigate and analyze chemical space

### Key Features

- 🧪 **Chemistry-Aware ML**: Integration with RDKit and DeepChem
- 🔗 **Graph Neural Networks**: GCN, GAT, GIN, and MPNN implementations
- 🎯 **Multi-task Learning**: Simultaneous prediction of multiple properties
- 🔄 **Molecular Generation**: VAE and GAN-based molecular generation
- 📊 **Experiment Tracking**: Weights & Biases integration
- 🔧 **Hyperparameter Optimization**: Optuna-based optimization
- 🌐 **REST API**: FastAPI-based web service
- 📈 **Model Interpretability**: SHAP integration for explainable AI

## 📁 Project Structure

```
Drug-Discovery-Compound-Optimization/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── data_processing/   # Data preprocessing and feature extraction package
│   │   ├── __init__.py
│   │   ├── core.py        # Shared imports, constants, and utilities
│   │   ├── processor.py   # MolecularDataProcessor (legacy interface)
│   │   ├── loader.py      # MolecularDataLoader for file handling
│   │   ├── preprocessor.py # MolecularPreprocessor for data cleaning
│   │   ├── feature_engineering.py # FeatureEnginerator for feature extraction
│   │   ├── data_splitting.py # DataSplitter for train/val/test splits
│   │   ├── splitting_strategies.py # Advanced splitting strategies
│   │   └── advanced_features.py # Advanced feature extraction methods
│   ├── logging_config.py  # Logging configuration
│   ├── models.py          # ML model implementations
│   ├── training.py        # Training loops and utilities
│   ├── api.py            # FastAPI web service
│   └── utils.py          # Utility functions
├── data/                  # Data storage
│   ├── raw/              # Raw datasets
│   ├── processed/        # Preprocessed data
│   ├── external/         # External reference data
│   └── test_sample.csv   # Sample test data
├── models/               # Model storage
│   ├── saved/           # Trained models
│   └── checkpoints/     # Training checkpoints
├── notebooks/           # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb # Comprehensive data exploration
│   └── 02_feature_engineering.ipynb # Advanced feature engineering
├── tests/              # Unit tests
│   ├── test_data_processing.py # Data processing tests
│   └── test_molecular_features.py # Molecular features tests
├── docs/               # Documentation
│   └── DATA_PIPELINE_SUMMARY.md # Pipeline implementation summary
├── config/             # Configuration files
│   ├── config.yaml     # Main configuration
│   ├── model_config.yaml  # Model parameters
│   └── data_config.yaml   # Data processing settings
├── scripts/            # Utility scripts
│   ├── process_data.py    # Complete data processing pipeline
│   ├── download_chembl.py # ChEMBL data downloader
│   ├── download_pubchem.py # PubChem data downloader
│   └── download_tox21.py  # Tox21 data downloader
├── logs/               # Log files directory
├── requirements.txt    # Python dependencies
├── setup_env.sh       # Linux/Mac environment setup script
├── setup_env.bat      # Windows environment setup script
├── install_pip.bat    # Pip-only installation script
├── INSTALLATION_COMPLETE.md # Installation completion guide
└── README.md          # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

#### Option 1: Automated Installation (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Drug-Discovery-Compound-Optimization.git
   cd Drug-Discovery-Compound-Optimization
   ```

2. **Run the installation script**
   
   **Windows:**
   ```cmd
   install_pip.bat
   ```
   
   **Linux/Mac:**
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```

3. **Verify installation**
   ```bash
   python -c "import torch, pandas, numpy; print('Core packages installed successfully!')"
   ```

#### Option 2: Manual Installation

If you prefer manual installation or don't have conda:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all requirements
pip install -r requirements.txt
```

## 💻 Usage Examples

### 1. Data Processing

#### Basic Processing (Legacy Interface)
```python
from src.data_processing import MolecularDataProcessor

# Initialize processor
processor = MolecularDataProcessor(config_path="config/data_config.yaml")

# Load and process SMILES data
smiles_data = ["CCO", "CC(=O)O", "c1ccccc1"]
processed_data = processor.process_smiles(smiles_data)

# Extract molecular features
features = processor.extract_features(processed_data)
```

#### Modular Processing (New Structure)
```python
from src.data_processing import (
    MolecularDataLoader, MolecularPreprocessor, 
    FeatureEnginerator, DataSplitter
)

# Load data from file
loader = MolecularDataLoader()
df = loader.load_csv_file("data/raw/molecules.csv")

# Clean and validate data
preprocessor = MolecularPreprocessor()
df_clean = preprocessor.validate_molecules(df)
df_clean = preprocessor.standardize_molecules(df_clean)

# Extract features
feature_eng = FeatureEnginerator()
df_features = feature_eng.extract_molecular_descriptors(df_clean)
df_features = feature_eng.extract_molecular_fingerprints(df_features)

# Split data
splitter = DataSplitter()
train_df, val_df, test_df = splitter.random_split(df_features, target_column='activity')
```

### 2. Model Training

```python
from src.models import GraphNeuralNetwork
from src.training import Trainer
import numpy as np

# Initialize model
model = GraphNeuralNetwork(
    model_type="gcn",
    hidden_dim=128,
    num_layers=3
)

# Create sample training data
train_data = np.random.randn(100, 50)  # 100 samples, 50 features
val_data = np.random.randn(20, 50)     # 20 validation samples

# Train model
trainer = Trainer(model, config_path="config/config.yaml")
trainer.train(train_data, val_data)
```

### 3. Property Prediction

```python
from src.models import PropertyPredictor

# Load trained model
predictor = PropertyPredictor.load("models/saved/property_predictor.pt")

# Predict properties
smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
properties = predictor.predict(smiles)
print(f"LogP: {properties['logp']:.2f}")
print(f"Solubility: {properties['solubility']:.2f}")
```

### 4. REST API

Start the API server:
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

Make predictions via API:
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"smiles": "CCO", "properties": ["logp", "solubility"]}
)
print(response.json())
```

### 5. Jupyter Notebooks

Launch JupyterLab for interactive exploration:
```bash
jupyter lab
```

Example notebooks:
- `notebooks/01_data_exploration.ipynb` - Data analysis and visualization
- `notebooks/02_model_training.ipynb` - Model training and evaluation
- `notebooks/03_compound_optimization.ipynb` - Molecular optimization

## 🔧 Configuration

The system uses YAML configuration files:

- **`config/config.yaml`**: Main configuration (paths, training settings, API settings)
- **`config/model_config.yaml`**: Model architectures and hyperparameters
- **`config/data_config.yaml`**: Data processing and feature extraction settings

Example configuration:
```yaml
# config/config.yaml
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100

models:
  save_path: "models/saved"
  
api:
  host: "0.0.0.0"
  port: 8000
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

## 📊 Experiment Tracking

### Weights & Biases Integration

1. **Setup W&B account**
   ```bash
   wandb login
   ```

2. **Enable in configuration**
   ```yaml
   # config/config.yaml
   wandb:
     project: "drug-discovery-optimization"
     enabled: true
   ```

3. **Track experiments**
   ```python
   from src.training import Trainer
   from src.models import PropertyPredictor
   import numpy as np
   
   # Initialize model
   model = PropertyPredictor()
   
   # Create sample training data
   train_data = np.random.randn(100, 50)  # 100 samples, 50 features  
   val_data = np.random.randn(20, 50)     # 20 validation samples
   
   # Initialize trainer with W&B tracking
   trainer = Trainer(model, use_wandb=True)
   trainer.train(train_data, val_data)
   ```

## 🔍 Model Interpretability

Use SHAP for model explanations:
```python
from src.utils import explain_prediction
from src.models import PropertyPredictor

# Load or create a model
model = PropertyPredictor()

# Explain a prediction
explanation = explain_prediction(model, smiles="CCO")
explanation.plot()
```

## 🚀 GPU Setup (Optional)

For GPU acceleration:

1. **Check CUDA version**
   ```bash
   nvidia-smi
   ```

2. **Install GPU PyTorch**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install additional GPU packages**
   ```bash
   pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
   ```

## 📚 Documentation

- **API Documentation**: Start the server and visit `http://localhost:8000/docs`
- **Code Documentation**: Generated with Sphinx (coming soon)
- **Tutorials**: See `notebooks/` directory

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [RDKit](https://www.rdkit.org/) - Cheminformatics toolkit
- [DeepChem](https://deepchem.io/) - Deep learning for chemistry
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Weights & Biases](https://wandb.ai/) - Experiment tracking

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/Drug-Discovery-Compound-Optimization/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/Drug-Discovery-Compound-Optimization/discussions)
- **Email**: your-email@example.com

---

**Happy Drug Discovery! 🧬💊**