"""
MolecularPreprocessor for molecular data cleaning and validation.

This module handles molecular data preprocessing including validation,
standardization, duplicate removal, and quality filtering.
"""

from typing import Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from .core import RDKIT_AVAILABLE, Chem, SaltRemover, logger


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
            df = df.copy()
            df['valid'] = True
            return df

        df = df.copy()

        # Handle empty DataFrame
        if len(df) == 0:
            df['valid'] = pd.Series([], dtype=bool)
            df['canonical_smiles'] = pd.Series([], dtype=object)
            return df

        # Handle missing SMILES column
        if smiles_column not in df.columns:
            logger.warning(f"SMILES column '{smiles_column}' not found in DataFrame")
            df['valid'] = False
            df['canonical_smiles'] = None
            return df

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

    def clean_smiles_column(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Clean SMILES column by removing invalid entries and standardizing format.

        Args:
            df: DataFrame with SMILES data
            smiles_column: Name of SMILES column

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()

        # Remove null/empty SMILES
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=[smiles_column])
        df_clean = df_clean[df_clean[smiles_column].str.strip() != '']
        
        null_removed = initial_count - len(df_clean)
        if null_removed > 0:
            logger.info(f"Removed {null_removed} rows with null/empty SMILES")

        # Remove duplicates based on SMILES
        if smiles_column in df_clean.columns:
            initial_count = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=[smiles_column])
            duplicates_removed = initial_count - len(df_clean)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate SMILES")

        return df_clean

    def filter_by_molecular_properties(self, df: pd.DataFrame, 
                                     property_filters: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Filter molecules based on calculated molecular properties.

        Args:
            df: DataFrame with molecular properties
            property_filters: Dictionary of property filters
                            Example: {'molecular_weight': {'min': 150, 'max': 500}}

        Returns:
            Filtered DataFrame
        """
        df_filtered = df.copy()
        initial_count = len(df_filtered)

        for property_name, criteria in property_filters.items():
            if property_name not in df_filtered.columns:
                logger.warning(f"Property {property_name} not found in DataFrame")
                continue

            if 'min' in criteria:
                before_count = len(df_filtered)
                df_filtered = df_filtered[df_filtered[property_name] >= criteria['min']]
                removed = before_count - len(df_filtered)
                if removed > 0:
                    logger.info(f"Removed {removed} molecules below {property_name} minimum ({criteria['min']})")

            if 'max' in criteria:
                before_count = len(df_filtered)
                df_filtered = df_filtered[df_filtered[property_name] <= criteria['max']]
                removed = before_count - len(df_filtered)
                if removed > 0:
                    logger.info(f"Removed {removed} molecules above {property_name} maximum ({criteria['max']})")

        total_removed = initial_count - len(df_filtered)
        logger.info(f"Property filtering removed {total_removed} total molecules ({initial_count} -> {len(df_filtered)})")
        
        return df_filtered

    def apply_lipinski_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Lipinski's Rule of Five filters.

        Args:
            df: DataFrame with molecular descriptors

        Returns:
            Filtered DataFrame
        """
        lipinski_filters = {
            'molecular_weight': {'min': 150, 'max': 500},
            'logp': {'min': -3, 'max': 5},
            'num_hbd': {'max': 5},
            'num_hba': {'max': 10},
            'tpsa': {'max': 140}
        }

        return self.filter_by_molecular_properties(df, lipinski_filters)

    def get_preprocessing_stats(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate preprocessing statistics.

        Args:
            original_df: Original DataFrame before preprocessing
            processed_df: DataFrame after preprocessing

        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            'original_count': len(original_df),
            'processed_count': len(processed_df),
            'removed_count': len(original_df) - len(processed_df),
            'retention_rate': len(processed_df) / len(original_df) * 100 if len(original_df) > 0 else 0,
            'columns_original': list(original_df.columns),
            'columns_processed': list(processed_df.columns),
            'new_columns': [col for col in processed_df.columns if col not in original_df.columns],
            'removed_columns': [col for col in original_df.columns if col not in processed_df.columns]
        }

        return stats