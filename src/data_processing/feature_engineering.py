"""
FeatureEnginerator for molecular feature extraction.

This module handles molecular feature extraction including descriptors
and fingerprints using RDKit with caching support.
"""

from typing import Dict, Any, Optional
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm

from .core import (
    RDKIT_AVAILABLE, Chem, Descriptors, rdMolDescriptors,
    GetMorganFingerprintAsBitVect, GetMorganGenerator, 
    MORGAN_GENERATOR_AVAILABLE, generate_cache_key, logger
)


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
            cache_file = self.cache_dir / f"descriptors_{generate_cache_key(df, smiles_column)}.pkl"
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
            cache_file = self.cache_dir / f"descriptors_{generate_cache_key(df, smiles_column)}.pkl"
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
            cache_file = self.cache_dir / f"fingerprints_{generate_cache_key(df, smiles_column)}.pkl"
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
                    # Morgan fingerprint - use new MorganGenerator if available
                    if fingerprints_config.get('morgan', {}).get('enabled', True):
                        morgan_config = fingerprints_config.get('morgan', {})
                        radius = morgan_config.get('radius', 2)
                        n_bits = morgan_config.get('n_bits', 2048)
                        if MORGAN_GENERATOR_AVAILABLE:
                            try:
                                morgan_gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
                                morgan_fp = morgan_gen.GetFingerprint(mol)
                                fingerprints['morgan_fp'] = list(morgan_fp.ToBitString())
                            except Exception:
                                # Fallback to old method
                                morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                                fingerprints['morgan_fp'] = list(morgan_fp.ToBitString())
                        else:
                            # Use old method
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
            cache_file = self.cache_dir / f"fingerprints_{generate_cache_key(df, smiles_column)}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(fingerprints_df, f)
            except Exception as e:
                logger.warning(f"Could not cache fingerprints: {e}")

        return pd.concat([df, fingerprints_df], axis=1)

    def get_advanced_features(self):
        """Get access to advanced feature extraction methods."""
        from .advanced_features import AdvancedFeatureExtractor
        return AdvancedFeatureExtractor(self.config)