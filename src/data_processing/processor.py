"""
MolecularDataProcessor for handling SMILES processing and feature extraction.

This module contains the legacy MolecularDataProcessor class that provides
a high-level interface for molecular data processing.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path

from .core import (
    RDKIT_AVAILABLE, Chem, Descriptors, GetMorganFingerprintAsBitVect,
    GetMorganGenerator, MORGAN_GENERATOR_AVAILABLE, load_config_from_path, logger
)


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
        self.config = load_config_from_path(config_path)
        self.descriptors = []
        self.fingerprints = []
        
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
            # Morgan fingerprint - use new MorganGenerator if available
            if MORGAN_GENERATOR_AVAILABLE:
                try:
                    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
                    morgan_fp = morgan_gen.GetFingerprint(mol)
                    fingerprints['morgan_fp'] = list(morgan_fp.ToBitString())
                except Exception:
                    # Fallback to old method
                    morgan_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fingerprints['morgan_fp'] = list(morgan_fp.ToBitString())
            else:
                # Use old method
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