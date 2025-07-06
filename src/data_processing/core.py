"""
Core module for data processing package.

Contains shared imports, constants, and utility classes used across the package.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# RDKit imports and availability check
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, PandasTools
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.SaltRemover import SaltRemover
    # Try to import newer MorganGenerator, fallback to old method
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        MORGAN_GENERATOR_AVAILABLE = True
    except ImportError:
        MORGAN_GENERATOR_AVAILABLE = False
        # Define dummy GetMorganGenerator if not available
        def GetMorganGenerator(radius=2, fpSize=2048):
            return None
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    MORGAN_GENERATOR_AVAILABLE = False
    # Define dummy classes for when RDKit is not available
    class Chem:
        @staticmethod
        def MolFromSmiles(smiles): return None
        @staticmethod
        def MolToSmiles(mol): return ""
        @staticmethod
        def RDKFingerprint(mol): return None
    
    class Descriptors:
        @staticmethod
        def MolWt(mol): return 0.0
        @staticmethod
        def MolLogP(mol): return 0.0
        @staticmethod
        def TPSA(mol): return 0.0
        @staticmethod
        def NumRotatableBonds(mol): return 0
        @staticmethod
        def NumHDonors(mol): return 0
        @staticmethod
        def NumHAcceptors(mol): return 0
        @staticmethod
        def NumAromaticRings(mol): return 0
        @staticmethod
        def RingCount(mol): return 0
        @staticmethod
        def MolMR(mol): return 0.0
        @staticmethod
        def NumSaturatedRings(mol): return 0
        @staticmethod
        def NumAliphaticRings(mol): return 0
        @staticmethod
        def BalabanJ(mol): return 0.0
        @staticmethod
        def BertzCT(mol): return 0.0
    
    class rdMolDescriptors:
        @staticmethod
        def GetMACCSKeysFingerprint(mol): return None
    
    class PandasTools:
        @staticmethod
        def LoadSDF(path): return pd.DataFrame()
    
    class MurckoScaffold:
        @staticmethod
        def MurckoScaffoldSmiles(mol, includeChirality=False): return ""
    
    class SaltRemover:
        def StripMol(self, mol): return mol
    
    def GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048): return None

    # Dummy MorganGenerator class and fingerprint objects
    class DummyFingerprint:
        def ToBitString(self):
            return "0" * 2048

    class DummyMorganGenerator:
        def __init__(self, radius=2, fpSize=2048):
            self.fpSize = fpSize
        def GetFingerprint(self, mol):
            return DummyFingerprint()

    def GetMorganGenerator(radius=2, fpSize=2048):
        return DummyMorganGenerator(radius, fpSize)

    logging.warning("RDKit not available. Some functionality will be limited.")

# Scikit-learn imports and availability check
try:
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define dummy functions for when sklearn is not available
    def train_test_split(X, y=None, test_size=None, stratify=None, random_state=None):
        if y is not None:
            return X, X, y, y
        else:
            return X, X
    
    class KMeans:
        def __init__(self, *args, **kwargs): pass
        def fit_predict(self, X): return np.zeros(len(X))
    
    class StratifiedKFold:
        def __init__(self, *args, **kwargs): pass
    
    logging.warning("Scikit-learn not available. Some functionality will be limited.")

# Module logger
logger = logging.getLogger(__name__)


def load_config_from_path(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file with fallback to defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'preprocessing': {
                'standardization': {
                    'remove_salts': True,
                    'neutralize_charges': True,
                    'canonicalize_tautomers': True
                }
            },
            'feature_extraction': {
                'descriptors': {
                    'rdkit_2d': {'enabled': True}
                },
                'fingerprints': {
                    'morgan': {'enabled': True, 'radius': 2, 'n_bits': 2048}
                }
            }
        }


def generate_cache_key(df: pd.DataFrame, smiles_column: str) -> str:
    """Generate cache key based on SMILES data."""
    smiles_str = ''.join(df[smiles_column].astype(str).tolist())
    return hashlib.md5(smiles_str.encode()).hexdigest()[:8]


def validate_file_format(file_path: str, supported_formats: List[str]) -> bool:
    """Validate if file format is supported."""
    return Path(file_path).suffix.lower() in supported_formats