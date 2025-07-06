"""
Data Processing Module for Drug Discovery Compound Optimization

This module handles molecular data preprocessing, feature extraction,
and data preparation for machine learning models.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import pickle
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, PandasTools
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem.SaltRemover import SaltRemover
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
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
    logging.warning("RDKit not available. Some functionality will be limited.")

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

logger = logging.getLogger(__name__)


class MolecularDataProcessor:
    """
    Handles molecular data processing including SMILES parsing,
    molecular descriptor calculation, and feature extraction.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the molecular data processor.
        
        Args:
            config_path: Path to data configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.descriptors = []
        self.fingerprints = []
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
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
    
    def process_smiles(self, smiles_list: List[str]) -> List[Dict[str, Any]]:
        """
        Process a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            List of processed molecular data dictionaries
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMILES processing")
            
        processed_data = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Standardize molecule if configured
                    if self.config.get('preprocessing', {}).get('standardization', {}).get('remove_salts', False):
                        mol = self._remove_salts(mol)
                    
                    processed_data.append({
                        'smiles': smiles,
                        'canonical_smiles': Chem.MolToSmiles(mol),
                        'mol': mol,
                        'valid': True
                    })
                else:
                    processed_data.append({
                        'smiles': smiles,
                        'canonical_smiles': None,
                        'mol': None,
                        'valid': False
                    })
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                processed_data.append({
                    'smiles': smiles,
                    'canonical_smiles': None,
                    'mol': None,
                    'valid': False
                })
        
        return processed_data
    
    def extract_features(self, processed_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Extract molecular features from processed data.
        
        Args:
            processed_data: List of processed molecular data
            
        Returns:
            DataFrame with molecular features
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for feature extraction")
            
        features_list = []
        
        for data in processed_data:
            if data['valid'] and data['mol'] is not None:
                features = self._calculate_descriptors(data['mol'])
                features.update(self._calculate_fingerprints(data['mol']))
                features['smiles'] = data['smiles']
                features['canonical_smiles'] = data['canonical_smiles']
            else:
                # Create empty feature dict for invalid molecules
                features = {'smiles': data['smiles'], 'canonical_smiles': data['canonical_smiles']}
                
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _calculate_descriptors(self, mol) -> Dict[str, float]:
        """Calculate molecular descriptors."""
        descriptors = {}
        
        try:
            descriptors['molecular_weight'] = Descriptors.MolWt(mol)
            descriptors['logp'] = Descriptors.MolLogP(mol)
            descriptors['tpsa'] = Descriptors.TPSA(mol)
            descriptors['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['num_hbd'] = Descriptors.NumHDonors(mol)
            descriptors['num_hba'] = Descriptors.NumHAcceptors(mol)
            descriptors['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
            descriptors['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
        except Exception as e:
            logger.warning(f"Error calculating descriptors: {e}")
            
        return descriptors
    
    def _calculate_fingerprints(self, mol) -> Dict[str, Any]:
        """Calculate molecular fingerprints."""
        fingerprints = {}
        
        try:
            # Morgan fingerprint
            morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints['morgan_fp'] = list(morgan_fp.ToBitString())
        except Exception as e:
            logger.warning(f"Error calculating fingerprints: {e}")
            
        return fingerprints
    
    def _remove_salts(self, mol):
        """Remove salts from molecule (placeholder implementation)."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated salt removal
        return mol
    
    def load_data(self, file_path: str, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Load molecular data from file.
        
        Args:
            file_path: Path to data file
            smiles_column: Name of SMILES column
            
        Returns:
            DataFrame with loaded data
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def save_processed_data(self, data: pd.DataFrame, output_path: str):
        """Save processed data to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.csv':
            data.to_csv(output_path, index=False)
        elif output_path.suffix.lower() in ['.xlsx', '.xls']:
            data.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")


class MolecularDataLoader:
    """
    Handles loading molecular data from various file formats including SMILES, SDF, and CSV.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the molecular data loader.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def load_smiles_file(self, file_path: str, smiles_column: str = 'smiles',
                        delimiter: str = '\t') -> pd.DataFrame:
        """
        Load SMILES data from text file.

        Args:
            file_path: Path to SMILES file
            smiles_column: Name of SMILES column
            delimiter: File delimiter

        Returns:
            DataFrame with SMILES data
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.smi':
            # Standard SMILES format: SMILES\tID
            df = pd.read_csv(file_path, sep=delimiter, header=None,
                           names=[smiles_column, 'id'])
        else:
            df = pd.read_csv(file_path, sep=delimiter)

        logger.info(f"Loaded {len(df)} molecules from {file_path}")
        return df

    def load_sdf_file(self, file_path: str) -> pd.DataFrame:
        """
        Load molecular data from SDF file.

        Args:
            file_path: Path to SDF file

        Returns:
            DataFrame with molecular data and properties
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SDF file loading")

        file_path = Path(file_path)

        try:
            # Load SDF using RDKit PandasTools
            df = PandasTools.LoadSDF(str(file_path))
            logger.info(f"Loaded {len(df)} molecules from SDF file {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading SDF file {file_path}: {e}")
            raise

    def load_csv_file(self, file_path: str, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Load molecular data from CSV file.

        Args:
            file_path: Path to CSV file
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with molecular data
        """
        file_path = Path(file_path)

        try:
            df = pd.read_csv(file_path)

            if smiles_column not in df.columns:
                raise ValueError(f"SMILES column '{smiles_column}' not found in CSV file")

            logger.info(f"Loaded {len(df)} molecules from CSV file {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            raise

    def load_excel_file(self, file_path: str, smiles_column: str = 'smiles',
                       sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """
        Load molecular data from Excel file.

        Args:
            file_path: Path to Excel file
            smiles_column: Name of SMILES column
            sheet_name: Sheet name or index

        Returns:
            DataFrame with molecular data
        """
        file_path = Path(file_path)

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            if smiles_column not in df.columns:
                raise ValueError(f"SMILES column '{smiles_column}' not found in Excel file")

            logger.info(f"Loaded {len(df)} molecules from Excel file {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {e}")
            raise

    def auto_load(self, file_path: str, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Automatically detect file format and load data.

        Args:
            file_path: Path to data file
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with molecular data
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        if suffix == '.sdf':
            return self.load_sdf_file(str(file_path))
        elif suffix == '.csv':
            return self.load_csv_file(str(file_path), smiles_column)
        elif suffix in ['.xlsx', '.xls']:
            return self.load_excel_file(str(file_path), smiles_column)
        elif suffix in ['.smi', '.txt']:
            return self.load_smiles_file(str(file_path), smiles_column)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


class MolecularPreprocessor:
    """
    Handles molecular data preprocessing including cleaning and validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the molecular preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        if RDKIT_AVAILABLE:
            self.salt_remover = SaltRemover()

    def validate_molecules(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Validate SMILES strings and add validity flags.

        Args:
            df: DataFrame with SMILES data
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with validity flags
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping molecule validation")
            df['valid'] = True
            return df

        df = df.copy()
        valid_flags = []
        canonical_smiles = []

        for smiles in tqdm(df[smiles_column], desc="Validating molecules"):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    valid_flags.append(True)
                    canonical_smiles.append(Chem.MolToSmiles(mol))
                else:
                    valid_flags.append(False)
                    canonical_smiles.append(None)
            except Exception:
                valid_flags.append(False)
                canonical_smiles.append(None)

        df['valid'] = valid_flags
        df['canonical_smiles'] = canonical_smiles

        logger.info(f"Validated {sum(valid_flags)}/{len(df)} molecules")
        return df

    def standardize_molecules(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Standardize molecules by removing salts, neutralizing charges, etc.

        Args:
            df: DataFrame with SMILES data
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with standardized molecules
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping molecule standardization")
            return df

        df = df.copy()
        standardized_smiles = []

        for smiles in tqdm(df[smiles_column], desc="Standardizing molecules"):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    # Remove salts
                    if self.config.get('preprocessing', {}).get('standardization', {}).get('remove_salts', True):
                        mol = self.salt_remover.StripMol(mol)

                    # Neutralize charges (simplified)
                    if self.config.get('preprocessing', {}).get('standardization', {}).get('neutralize_charges', True):
                        mol = self._neutralize_charges(mol)

                    standardized_smiles.append(Chem.MolToSmiles(mol))
                else:
                    standardized_smiles.append(smiles)
            except Exception as e:
                logger.warning(f"Error standardizing {smiles}: {e}")
                standardized_smiles.append(smiles)

        df['standardized_smiles'] = standardized_smiles
        return df

    def _neutralize_charges(self, mol):
        """Neutralize charges in molecule (simplified implementation)."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated charge neutralization
        return mol

    def remove_duplicates(self, df: pd.DataFrame, smiles_column: str = 'canonical_smiles',
                         method: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate molecules.

        Args:
            df: DataFrame with molecular data
            smiles_column: Column to use for duplicate detection
            method: Method for handling duplicates ('first', 'last', 'random')

        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)

        if method == 'random':
            df = df.sample(frac=1).reset_index(drop=True)

        df_dedup = df.drop_duplicates(subset=[smiles_column], keep=method)

        logger.info(f"Removed {initial_count - len(df_dedup)} duplicates")
        return df_dedup

    def apply_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quality filters based on molecular properties.

        Args:
            df: DataFrame with molecular data and descriptors

        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()
        initial_count = len(df_filtered)

        # Apply Lipinski's Rule of Five filters
        filters = self.config.get('preprocessing', {}).get('quality_filters', {})

        for prop, criteria in filters.items():
            if prop in df_filtered.columns:
                if 'min' in criteria:
                    df_filtered = df_filtered[df_filtered[prop] >= criteria['min']]
                if 'max' in criteria:
                    df_filtered = df_filtered[df_filtered[prop] <= criteria['max']]

        logger.info(f"Quality filters removed {initial_count - len(df_filtered)} molecules")
        return df_filtered


class FeatureEnginerator:
    """
    Handles molecular feature extraction using RDKit.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the feature enginerator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.cache_dir = Path(self.config.get('caching', {}).get('cache_dir', 'data/cache'))
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create cache directory {self.cache_dir}: {e}")
            # Use a temporary directory or disable caching
            self.cache_dir = None

    def extract_molecular_descriptors(self, df: pd.DataFrame,
                                    smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Extract molecular descriptors using RDKit.

        Args:
            df: DataFrame with SMILES data
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with molecular descriptors
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping descriptor calculation")
            return df

        df = df.copy()

        # Check cache
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"descriptors_{self._get_cache_key(df, smiles_column)}.pkl"
            if cache_file.exists() and self.config.get('caching', {}).get('cache_descriptors', True):
                logger.info("Loading descriptors from cache")
                with open(cache_file, 'rb') as f:
                    descriptors_df = pickle.load(f)
                return pd.concat([df, descriptors_df], axis=1)

        descriptors_list = []

        for smiles in tqdm(df[smiles_column], desc="Calculating descriptors"):
            descriptors = {}
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    # Basic descriptors
                    descriptors['molecular_weight'] = Descriptors.MolWt(mol)
                    descriptors['logp'] = Descriptors.MolLogP(mol)
                    descriptors['tpsa'] = Descriptors.TPSA(mol)
                    descriptors['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                    descriptors['num_hbd'] = Descriptors.NumHDonors(mol)
                    descriptors['num_hba'] = Descriptors.NumHAcceptors(mol)
                    descriptors['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
                    descriptors['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
                    descriptors['num_rings'] = Descriptors.RingCount(mol)
                    descriptors['molar_refractivity'] = Descriptors.MolMR(mol)

                    # Additional descriptors
                    descriptors['num_saturated_rings'] = Descriptors.NumSaturatedRings(mol)
                    descriptors['num_aliphatic_rings'] = Descriptors.NumAliphaticRings(mol)
                    descriptors['balaban_j'] = Descriptors.BalabanJ(mol)
                    descriptors['bertz_ct'] = Descriptors.BertzCT(mol)

            except Exception as e:
                logger.warning(f"Error calculating descriptors for {smiles}: {e}")

            descriptors_list.append(descriptors)

        descriptors_df = pd.DataFrame(descriptors_list)

        # Cache descriptors
        if self.cache_dir is not None and self.config.get('caching', {}).get('cache_descriptors', True):
            cache_file = self.cache_dir / f"descriptors_{self._get_cache_key(df, smiles_column)}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(descriptors_df, f)
            except Exception as e:
                logger.warning(f"Could not cache descriptors: {e}")

        return pd.concat([df, descriptors_df], axis=1)

    def extract_molecular_fingerprints(self, df: pd.DataFrame,
                                     smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Extract molecular fingerprints using RDKit.

        Args:
            df: DataFrame with SMILES data
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with molecular fingerprints
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping fingerprint calculation")
            return df

        df = df.copy()

        # Check cache
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"fingerprints_{self._get_cache_key(df, smiles_column)}.pkl"
            if cache_file.exists() and self.config.get('caching', {}).get('cache_fingerprints', True):
                logger.info("Loading fingerprints from cache")
                with open(cache_file, 'rb') as f:
                    fingerprints_df = pickle.load(f)
                return pd.concat([df, fingerprints_df], axis=1)

        fingerprints_config = self.config.get('feature_extraction', {}).get('fingerprints', {})
        fingerprints_list = []

        for smiles in tqdm(df[smiles_column], desc="Calculating fingerprints"):
            fingerprints = {}
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    # Morgan fingerprint
                    if fingerprints_config.get('morgan', {}).get('enabled', True):
                        morgan_config = fingerprints_config.get('morgan', {})
                        radius = morgan_config.get('radius', 2)
                        n_bits = morgan_config.get('n_bits', 2048)
                        morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                        fingerprints['morgan_fp'] = list(morgan_fp.ToBitString())

                    # RDKit fingerprint
                    if fingerprints_config.get('rdkit', {}).get('enabled', True):
                        rdkit_fp = Chem.RDKFingerprint(mol)
                        fingerprints['rdkit_fp'] = list(rdkit_fp.ToBitString())

                    # MACCS keys
                    if fingerprints_config.get('maccs', {}).get('enabled', True):
                        maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                        fingerprints['maccs_fp'] = list(maccs_fp.ToBitString())

            except Exception as e:
                logger.warning(f"Error calculating fingerprints for {smiles}: {e}")

            fingerprints_list.append(fingerprints)

        fingerprints_df = pd.DataFrame(fingerprints_list)

        # Cache fingerprints
        if self.cache_dir is not None and self.config.get('caching', {}).get('cache_fingerprints', True):
            cache_file = self.cache_dir / f"fingerprints_{self._get_cache_key(df, smiles_column)}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(fingerprints_df, f)
            except Exception as e:
                logger.warning(f"Could not cache fingerprints: {e}")

        return pd.concat([df, fingerprints_df], axis=1)

    def _get_cache_key(self, df: pd.DataFrame, smiles_column: str) -> str:
        """Generate cache key based on SMILES data."""
        smiles_str = ''.join(df[smiles_column].astype(str).tolist())
        return hashlib.md5(smiles_str.encode()).hexdigest()[:8]


class DataSplitter:
    """
    Handles train/validation/test splits with stratification and scaffold-based splitting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data splitter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def random_split(self, df: pd.DataFrame, target_column: Optional[str] = None,
                    train_ratio: float = 0.7, val_ratio: float = 0.15,
                    test_ratio: float = 0.15, stratify: bool = False,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform random train/validation/test split.

        Args:
            df: DataFrame to split
            target_column: Target column for stratification
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify: Whether to stratify split
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for data splitting")

        # Normalize ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

        # First split: separate test set
        if stratify and target_column and target_column in df.columns:
            train_val_df, test_df = train_test_split(
                df, test_size=test_ratio, stratify=df[target_column],
                random_state=random_state
            )
        else:
            train_val_df, test_df = train_test_split(
                df, test_size=test_ratio, random_state=random_state
            )

        # Second split: separate train and validation
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

        if stratify and target_column and target_column in train_val_df.columns:
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_ratio_adjusted,
                stratify=train_val_df[target_column], random_state=random_state
            )
        else:
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_ratio_adjusted, random_state=random_state
            )

        logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def scaffold_split(self, df: pd.DataFrame, smiles_column: str = 'canonical_smiles',
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform scaffold-based split for better generalization.

        Args:
            df: DataFrame to split
            smiles_column: SMILES column name
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for scaffold splitting")

        # Calculate scaffolds
        scaffolds = {}
        for idx, smiles in enumerate(df[smiles_column]):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                    if scaffold not in scaffolds:
                        scaffolds[scaffold] = []
                    scaffolds[scaffold].append(idx)
            except Exception as e:
                logger.warning(f"Error calculating scaffold for {smiles}: {e}")
                # Assign to a unique scaffold
                unique_scaffold = f"unique_{idx}"
                scaffolds[unique_scaffold] = [idx]

        # Sort scaffolds by size (largest first)
        scaffold_sets = list(scaffolds.values())
        scaffold_sets.sort(key=len, reverse=True)

        # Assign scaffolds to splits
        train_indices, val_indices, test_indices = [], [], []
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)

        for scaffold_set in scaffold_sets:
            if len(train_indices) < train_size:
                train_indices.extend(scaffold_set)
            elif len(val_indices) < val_size:
                val_indices.extend(scaffold_set)
            else:
                test_indices.extend(scaffold_set)

        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        logger.info(f"Scaffold split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def cluster_split(self, df: pd.DataFrame, features: List[str],
                     train_ratio: float = 0.7, val_ratio: float = 0.15,
                     test_ratio: float = 0.15, n_clusters: Optional[int] = None,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform cluster-based split.

        Args:
            df: DataFrame to split
            features: Feature columns for clustering
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            n_clusters: Number of clusters (auto if None)
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for cluster splitting")

        # Prepare feature matrix
        feature_matrix = df[features].fillna(0).values

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(50, len(df) // 10)  # Heuristic

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(feature_matrix)

        # Create cluster-based splits
        cluster_df = df.copy()
        cluster_df['cluster'] = clusters

        # Split clusters
        unique_clusters = np.unique(clusters)
        np.random.seed(random_state)
        np.random.shuffle(unique_clusters)

        n_train_clusters = int(len(unique_clusters) * train_ratio)
        n_val_clusters = int(len(unique_clusters) * val_ratio)

        train_clusters = unique_clusters[:n_train_clusters]
        val_clusters = unique_clusters[n_train_clusters:n_train_clusters + n_val_clusters]
        test_clusters = unique_clusters[n_train_clusters + n_val_clusters:]

        train_df = cluster_df[cluster_df['cluster'].isin(train_clusters)].drop('cluster', axis=1)
        val_df = cluster_df[cluster_df['cluster'].isin(val_clusters)].drop('cluster', axis=1)
        test_df = cluster_df[cluster_df['cluster'].isin(test_clusters)].drop('cluster', axis=1)

        logger.info(f"Cluster split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df


def main():
    """Example usage of the MolecularDataProcessor."""
    # Example SMILES data
    smiles_data = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "invalid_smiles"  # Invalid SMILES for testing
    ]
    
    # Initialize processor
    processor = MolecularDataProcessor()
    
    # Process SMILES
    processed_data = processor.process_smiles(smiles_data)
    print(f"Processed {len(processed_data)} molecules")
    
    # Extract features
    if RDKIT_AVAILABLE:
        features_df = processor.extract_features(processed_data)
        print(f"Extracted features for {len(features_df)} molecules")
        print(features_df.head())
    else:
        print("RDKit not available - skipping feature extraction")


if __name__ == "__main__":
    main()