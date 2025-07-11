# 🧬 Drug Discovery System - Production Summary

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
## ✅ Overview

A scalable and production-ready system for **molecular property prediction**, **compound optimization**, and **drug discovery** using:

- RESTful API (FastAPI)
- Data pipeline for molecular preprocessing
- GNN-based property prediction models
- Dockerized deployment
- Interactive web UI
- Comprehensive testing

---
## What This Project Does

This project is designed to help scientists discover new medicines faster and more safely. Imagine you have thousands of chemical compounds and need to figure out which ones are safe, effective, and worth testing further. Doing this by hand would take forever—but this system does it automatically.

It takes raw chemical data and turns it into clean, structured information. Then it uses artificial intelligence to predict how each compound might behave—whether it’s likely to be useful, toxic, or promising for research. Scientists can upload their data through a simple web interface and instantly get back detailed results, complete with easy-to-understand reports and visuals.

Everything is built to be fast, accurate, and ready to use in the real world. No technical setup needed—just upload your data and get insights.

## 🚀 Core Features

### 🧠 Machine Learning
- Property prediction (ADMET, bioactivity, toxicity)
- Compound optimization via multi-target objective
- GNN models: GCN, GAT, GIN, MPNN
- Feature extraction with RDKit & fingerprints

### ⚙️ API System
- FastAPI with Swagger UI
- SMILES validation, batch prediction, optimization endpoints
- Rate limiting (`slowapi`), Redis caching
- Health checks, metrics, error handling
- Dockerized with Nginx + Redis

### 💻 Web Interface
- Bootstrap UI with drag-drop upload
- Interactive results + visualization
- JavaScript frontend with live API integration

---

## 🛠 Data Pipeline

- Load SMILES/SDF/CSV/Excel
- Preprocessing: standardization, deduplication
- Feature engineering: descriptors, fingerprints, custom features
- Splitting strategies: random, scaffold, cluster
- CLI script & modular Python API

---

## 📦 Project Structure

```
Drug-Discovery-Compound-Optimization/
│
├── config/                          # Configuration files
│   ├── config.yaml                  # Main configuration
│   ├── data_config.yaml             # Data processing configuration
│   └── model_config.yaml            # Model configuration
│
├── src/                             # Main source code
│   ├── data_processing/             # Data processing modules
│   │   ├── __init__.py
│   │   ├── core.py                  # Core utilities and classes
│   │   ├── loader.py                # Data loading functionality
│   │   ├── preprocessor.py          # Data preprocessing
│   │   ├── processor.py             # Main data processor
│   │   ├── feature_engineering.py   # Feature extraction and engineering
│   │   ├── data_splitting.py        # Data splitting utilities
│   │   ├── splitting_strategies.py  # Advanced splitting strategies
│   │   └── advanced_features.py     # Advanced feature computations
│   │
│   ├── __init__.py
│   ├── api.py                       # Main API implementation
│   ├── api_models.py                # Pydantic models for API
│   ├── api_simple.py                # Simplified API version
│   ├── data_processing.py           # Data processing entry point
│   ├── logging_config.py            # Logging configuration
│   ├── models.py                    # Machine learning models
│   ├── training.py                  # Model training utilities
│   └── utils.py                     # General utility functions
│
├── scripts/                         # Utility and deployment scripts
│   ├── deploy_docker.py             # Docker deployment script
│   ├── process_data.py              # Data processing CLI script
│   ├── run_api.py                   # API startup script
│   ├── manual_chembl_download.py    # ChEMBL data downloader
│   ├── manual_pubchem_download.py   # PubChem data downloader
│   ├── manual_tox21_download.py     # Tox21 data downloader
│   └── .gitkeep
│
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── test_api.py                  # API endpoint tests
│   ├── test_data_processing.py      # Data processing tests
│   └── test_molecular_features.py   # Molecular feature tests
│
├── data/                            # Data storage
│   ├── raw/                         # Raw, unprocessed data
│   ├── processed/                   # Processed and cleaned data
│   └── cache/                       # Cached computational results
│
├── logs/                            # Log files
│
├── models/                          # Saved machine learning models
│   ├── checkpoints/                 # Model checkpoints
│   └── saved/                       # Saved trained models
│
├── static/                          # Static web assets
│   ├── css/                         # Stylesheets
│   ├── js/                          # JavaScript files
│   └── images/                      # Image assets
│
├── templates/                       # HTML templates
│
├── docs/                            # Project documentation
│
├── requirements.txt                 # Production dependencies
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Docker Compose configuration
├── docker-compose.dev.yml           # Development Docker Compose
├── setup_env.sh                     # Environment setup script (Unix)
├── setup_env.bat                    # Environment setup script (Windows)
├── install_pip.bat                  # Pip installation script (Windows)
├── package-lock.json                # Package lock file
├── LICENSE                          # License file
├── .gitignore                       # Git ignore rules
├── .gitattributes                   # Git attributes
└── README.md                        # This file
```

---

## 📈 Deployment

- Dockerized API (multi-stage build)
- Docker Compose with Redis, Nginx, dev tools
- CLI deploy script with health verification
- Config-driven (YAML) runtime setup

---

## 🏁 Summary

✅ Feature-complete and production-ready system:
- Robust API + GNN models
- Data pipeline with CLI + notebook support
- Deployment and monitoring built-in
- Tested, modular, and extensible

**Ready for real-world drug discovery workflows.**
