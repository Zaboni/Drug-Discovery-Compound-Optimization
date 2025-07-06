"""
Advanced feature extraction methods for molecular data.

This module contains extended feature extraction capabilities including
custom descriptors, feature importance analysis, and advanced fingerprints.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from tqdm import tqdm

from .core import RDKIT_AVAILABLE, Chem, Descriptors, logger


class AdvancedFeatureExtractor:
    """Advanced feature extraction methods for molecular data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced feature extractor."""
        self.config = config or {}

    def extract_custom_descriptors(self, df: pd.DataFrame, 
                                  smiles_column: str = 'canonical_smiles',
                                  descriptor_list: List = None) -> pd.DataFrame:
        """
        Extract custom molecular descriptors.

        Args:
            df: DataFrame with SMILES data
            smiles_column: Name of SMILES column
            descriptor_list: List of descriptor functions to calculate

        Returns:
            DataFrame with custom descriptors
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping custom descriptor calculation")
            return df

        if descriptor_list is None:
            descriptor_list = [
                ('molecular_weight', Descriptors.MolWt),
                ('logp', Descriptors.MolLogP),
                ('tpsa', Descriptors.TPSA),
                ('num_rotatable_bonds', Descriptors.NumRotatableBonds),
                ('num_hbd', Descriptors.NumHDonors),
                ('num_hba', Descriptors.NumHAcceptors)
            ]

        df = df.copy()
        custom_descriptors = []

        for smiles in tqdm(df[smiles_column], desc="Calculating custom descriptors"):
            descriptors = {}
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    for desc_name, desc_func in descriptor_list:
                        try:
                            descriptors[desc_name] = desc_func(mol)
                        except Exception as e:
                            logger.warning(f"Error calculating {desc_name} for {smiles}: {e}")
                            descriptors[desc_name] = None
            except Exception as e:
                logger.warning(f"Error processing molecule {smiles}: {e}")

            custom_descriptors.append(descriptors)

        custom_df = pd.DataFrame(custom_descriptors)
        return pd.concat([df, custom_df], axis=1)

    def get_available_descriptors(self) -> Dict[str, Any]:
        """
        Get list of available RDKit descriptors.

        Returns:
            Dictionary of available descriptors with descriptions
        """
        if not RDKIT_AVAILABLE:
            return {"error": "RDKit not available"}

        available_descriptors = {
            'basic': {
                'molecular_weight': 'Molecular weight in Daltons',
                'logp': 'Octanol-water partition coefficient',
                'tpsa': 'Topological polar surface area',
                'num_rotatable_bonds': 'Number of rotatable bonds',
                'num_hbd': 'Number of hydrogen bond donors',
                'num_hba': 'Number of hydrogen bond acceptors',
                'num_heavy_atoms': 'Number of heavy atoms',
                'num_aromatic_rings': 'Number of aromatic rings',
                'num_rings': 'Total number of rings'
            },
            'extended': {
                'molar_refractivity': 'Molar refractivity',
                'num_saturated_rings': 'Number of saturated rings',
                'num_aliphatic_rings': 'Number of aliphatic rings',
                'balaban_j': 'Balaban J index',
                'bertz_ct': 'Bertz CT descriptor'
            }
        }

        return available_descriptors

    def get_feature_importance(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, float]:
        """
        Calculate feature importance for molecular descriptors.

        Args:
            df: DataFrame with features and target
            target_column: Name of target column

        Returns:
            Dictionary with feature importance scores
        """
        if target_column is None or target_column not in df.columns:
            logger.warning("Target column not specified or not found")
            return {}

        # Get numeric descriptor columns
        descriptor_cols = [col for col in df.columns 
                          if col.startswith(('molecular_', 'logp', 'tpsa', 'num_', 'molar_', 'balaban_', 'bertz_'))]

        if not descriptor_cols:
            logger.warning("No descriptor columns found")
            return {}

        try:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.preprocessing import StandardScaler
            import numpy as np

            # Prepare data
            X = df[descriptor_cols].fillna(0)
            y = df[target_column].dropna()

            # Align X and y
            valid_indices = y.index
            X = X.loc[valid_indices]

            if len(X) == 0:
                return {}

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Calculate mutual information
            mi_scores = mutual_info_regression(X_scaled, y)

            # Create importance dictionary
            importance = dict(zip(descriptor_cols, mi_scores))
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            return importance

        except ImportError:
            logger.warning("Scikit-learn not available for feature importance calculation")
            return {}
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}

    def create_feature_matrix(self, df: pd.DataFrame, 
                            include_descriptors: bool = True,
                            include_fingerprints: bool = True,
                            smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Create a complete feature matrix from molecular data.

        Args:
            df: DataFrame with SMILES data
            include_descriptors: Whether to include molecular descriptors
            include_fingerprints: Whether to include fingerprints
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with complete feature matrix
        """
        from .feature_engineering import FeatureEnginerator
        
        feature_df = df.copy()
        feature_eng = FeatureEnginerator(self.config)

        if include_descriptors:
            logger.info("Extracting molecular descriptors")
            feature_df = feature_eng.extract_molecular_descriptors(feature_df, smiles_column)

        if include_fingerprints:
            logger.info("Extracting molecular fingerprints")
            feature_df = feature_eng.extract_molecular_fingerprints(feature_df, smiles_column)

        # Remove non-numeric columns except essential ones
        essential_columns = ['smiles', 'canonical_smiles', 'valid']
        numeric_columns = feature_df.select_dtypes(include=['number']).columns.tolist()
        keep_columns = essential_columns + numeric_columns
        keep_columns = [col for col in keep_columns if col in feature_df.columns]

        feature_df = feature_df[keep_columns]

        logger.info(f"Created feature matrix with {len(feature_df)} rows and {len(feature_df.columns)} columns")
        return feature_df

    def calculate_physicochemical_properties(self, df: pd.DataFrame,
                                           smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Calculate physicochemical properties for drug-like assessment.

        Args:
            df: DataFrame with SMILES data
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with physicochemical properties
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping physicochemical calculation")
            return df

        df = df.copy()
        properties_list = []

        for smiles in tqdm(df[smiles_column], desc="Calculating physicochemical properties"):
            properties = {}
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    # Lipinski descriptors
                    properties['mw'] = Descriptors.MolWt(mol)
                    properties['logp'] = Descriptors.MolLogP(mol)
                    properties['hbd'] = Descriptors.NumHDonors(mol)
                    properties['hba'] = Descriptors.NumHAcceptors(mol)
                    
                    # Additional drug-like properties
                    properties['tpsa'] = Descriptors.TPSA(mol)
                    properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                    properties['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
                    properties['fsp3'] = Descriptors.FractionCsp3(mol) if hasattr(Descriptors, 'FractionCsp3') else None
                    
                    # Lipinski violations
                    lipinski_violations = 0
                    if properties['mw'] > 500: lipinski_violations += 1
                    if properties['logp'] > 5: lipinski_violations += 1
                    if properties['hbd'] > 5: lipinski_violations += 1
                    if properties['hba'] > 10: lipinski_violations += 1
                    
                    properties['lipinski_violations'] = lipinski_violations
                    properties['lipinski_compliant'] = lipinski_violations <= 1

            except Exception as e:
                logger.warning(f"Error calculating properties for {smiles}: {e}")

            properties_list.append(properties)

        properties_df = pd.DataFrame(properties_list)
        return pd.concat([df, properties_df], axis=1)

    def calculate_admet_descriptors(self, df: pd.DataFrame,
                                   smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Calculate ADMET-related descriptors.

        Args:
            df: DataFrame with SMILES data
            smiles_column: Name of SMILES column

        Returns:
            DataFrame with ADMET descriptors
        """
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - skipping ADMET calculation")
            return df

        df = df.copy()
        admet_list = []

        for smiles in tqdm(df[smiles_column], desc="Calculating ADMET descriptors"):
            admet = {}
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    # Permeability-related
                    admet['tpsa'] = Descriptors.TPSA(mol)
                    admet['logp'] = Descriptors.MolLogP(mol)
                    
                    # Metabolism-related
                    admet['num_rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
                    admet['num_rings'] = Descriptors.RingCount(mol)
                    
                    # Toxicity-related predictors
                    admet['num_aromatic_rings'] = Descriptors.NumAromaticRings(mol)
                    admet['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
                    
                    # Blood-brain barrier penetration indicators
                    admet['mw'] = Descriptors.MolWt(mol)
                    admet['hbd'] = Descriptors.NumHDonors(mol)

            except Exception as e:
                logger.warning(f"Error calculating ADMET descriptors for {smiles}: {e}")

            admet_list.append(admet)

        admet_df = pd.DataFrame(admet_list)
        return pd.concat([df, admet_df], axis=1)