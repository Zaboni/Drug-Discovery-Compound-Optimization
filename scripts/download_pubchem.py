#!/usr/bin/env python3
"""
PubChem Data Download Script

This script downloads compound data from PubChem database.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    import pubchempy as pcp
    PUBCHEMPY_AVAILABLE = True
except ImportError:
    PUBCHEMPY_AVAILABLE = False
    # Define dummy functions for when not available
    class DummyCompound:
        def __init__(self):
            self.cid = 0
            self.canonical_smiles = ""
            self.molecular_formula = ""
            self.molecular_weight = 0.0
            self.xlogp = 0.0
            self.tpsa = 0.0
            self.h_bond_donor_count = 0
            self.h_bond_acceptor_count = 0
            self.rotatable_bond_count = 0
            self.heavy_atom_count = 0
            self.complexity = 0.0

        @classmethod
        def from_cid(cls, cid):
            return cls()

    class DummyPCP:
        Compound = DummyCompound

        @staticmethod
        def get_compounds(identifiers, namespace='name', **kwargs):
            return []
        @staticmethod
        def download(format_type, path, identifiers, operation=None):
            pass
    pcp = DummyPCP()
    print("PubChemPy not available. Install with: pip install pubchempy")

logger = logging.getLogger(__name__)


class PubChemDownloader:
    """Downloads data from PubChem database."""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize PubChem downloader.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
    def download_compounds_by_name(self, compound_names: list, 
                                  properties: list = None) -> pd.DataFrame:
        """
        Download compound data by compound names.
        
        Args:
            compound_names: List of compound names
            properties: List of properties to download
            
        Returns:
            DataFrame with compound data
        """
        if not PUBCHEMPY_AVAILABLE:
            raise ImportError("PubChemPy is required")
            
        if properties is None:
            properties = ['CanonicalSMILES', 'MolecularWeight', 'XLogP', 'TPSA', 
                         'HBondDonorCount', 'HBondAcceptorCount', 'RotatableBondCount']
        
        logger.info(f"Downloading data for {len(compound_names)} compounds")
        
        compounds_data = []
        for name in tqdm(compound_names, desc="Downloading compounds"):
            try:
                # Get compound by name
                compounds = pcp.get_compounds(name, 'name')
                
                if compounds:
                    compound = compounds[0]  # Take first match
                    
                    compound_data = {
                        'name': name,
                        'cid': compound.cid,
                        'canonical_smiles': compound.canonical_smiles,
                        'molecular_formula': compound.molecular_formula,
                        'molecular_weight': compound.molecular_weight,
                        'xlogp': compound.xlogp,
                        'tpsa': compound.tpsa,
                        'hbd': compound.h_bond_donor_count,
                        'hba': compound.h_bond_acceptor_count,
                        'rotatable_bonds': compound.rotatable_bond_count,
                        'heavy_atom_count': compound.heavy_atom_count,
                        'complexity': compound.complexity
                    }
                    
                    compounds_data.append(compound_data)
                    
                else:
                    logger.warning(f"No compound found for name: {name}")
                    
                time.sleep(0.2)  # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error downloading compound {name}: {e}")
        
        if compounds_data:
            df = pd.DataFrame(compounds_data)
            logger.info(f"Downloaded data for {len(df)} compounds")
            return df
        else:
            logger.warning("No compound data downloaded")
            return pd.DataFrame()
    
    def download_compounds_by_cid(self, cids: list) -> pd.DataFrame:
        """
        Download compound data by CIDs.
        
        Args:
            cids: List of PubChem CIDs
            
        Returns:
            DataFrame with compound data
        """
        if not PUBCHEMPY_AVAILABLE:
            raise ImportError("PubChemPy is required")
        
        logger.info(f"Downloading data for {len(cids)} CIDs")
        
        compounds_data = []
        for cid in tqdm(cids, desc="Downloading compounds"):
            try:
                compound = pcp.Compound.from_cid(cid)
                
                compound_data = {
                    'cid': compound.cid,
                    'canonical_smiles': compound.canonical_smiles,
                    'molecular_formula': compound.molecular_formula,
                    'molecular_weight': compound.molecular_weight,
                    'xlogp': compound.xlogp,
                    'tpsa': compound.tpsa,
                    'hbd': compound.h_bond_donor_count,
                    'hba': compound.h_bond_acceptor_count,
                    'rotatable_bonds': compound.rotatable_bond_count,
                    'heavy_atom_count': compound.heavy_atom_count,
                    'complexity': compound.complexity
                }
                
                compounds_data.append(compound_data)
                time.sleep(0.2)  # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error downloading CID {cid}: {e}")
        
        if compounds_data:
            df = pd.DataFrame(compounds_data)
            logger.info(f"Downloaded data for {len(df)} compounds")
            return df
        else:
            logger.warning("No compound data downloaded")
            return pd.DataFrame()
    
    def download_bioassay_data(self, aid: int, max_records: int = 10000) -> pd.DataFrame:
        """
        Download bioassay data.
        
        Args:
            aid: Assay ID
            max_records: Maximum number of records
            
        Returns:
            DataFrame with bioassay data
        """
        logger.info(f"Downloading bioassay data for AID {aid}")
        
        try:
            # Use REST API to get bioassay data
            url = f"{self.base_url}/assay/aid/{aid}/CSV"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Save to temporary file and read as CSV
                temp_file = self.output_dir / f"temp_aid_{aid}.csv"
                with open(temp_file, 'w') as f:
                    f.write(response.text)
                
                df = pd.read_csv(temp_file)
                temp_file.unlink()  # Remove temporary file
                
                # Limit records
                if len(df) > max_records:
                    df = df.head(max_records)
                
                logger.info(f"Downloaded {len(df)} bioassay records")
                return df
            else:
                logger.error(f"Failed to download bioassay data: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error downloading bioassay data: {e}")
            return pd.DataFrame()
    
    def search_compounds(self, query: str, max_results: int = 1000) -> pd.DataFrame:
        """
        Search for compounds using text query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            DataFrame with search results
        """
        if not PUBCHEMPY_AVAILABLE:
            raise ImportError("PubChemPy is required")
        
        logger.info(f"Searching for compounds with query: {query}")
        
        try:
            # Search compounds
            results = pcp.get_compounds(query, 'name', listkey_count=max_results)
            
            compounds_data = []
            for compound in tqdm(results[:max_results], desc="Processing search results"):
                try:
                    compound_data = {
                        'cid': compound.cid,
                        'canonical_smiles': compound.canonical_smiles,
                        'molecular_formula': compound.molecular_formula,
                        'molecular_weight': compound.molecular_weight,
                        'xlogp': compound.xlogp,
                        'tpsa': compound.tpsa,
                        'hbd': compound.h_bond_donor_count,
                        'hba': compound.h_bond_acceptor_count,
                        'rotatable_bonds': compound.rotatable_bond_count
                    }
                    compounds_data.append(compound_data)
                except Exception as e:
                    logger.warning(f"Error processing compound {compound.cid}: {e}")
            
            if compounds_data:
                df = pd.DataFrame(compounds_data)
                logger.info(f"Found {len(df)} compounds")
                return df
            else:
                logger.warning("No compounds found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error searching compounds: {e}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate downloaded data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_records': len(df),
            'missing_smiles': 0,
            'invalid_smiles': 0,
            'missing_values': {},
            'duplicate_records': 0
        }
        
        if 'canonical_smiles' in df.columns:
            validation_results['missing_smiles'] = df['canonical_smiles'].isnull().sum()
            
            # Check for invalid SMILES (basic check)
            try:
                from rdkit import Chem
                invalid_count = 0
                for smiles in df['canonical_smiles'].dropna():
                    mol = Chem.MolFromSmiles(str(smiles))
                    if mol is None:
                        invalid_count += 1
                validation_results['invalid_smiles'] = invalid_count
            except ImportError:
                # Define Chem as None when RDKit is not available
                Chem = None
                logger.warning("RDKit not available - skipping SMILES validation")
        
        # Check for missing values in each column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                validation_results['missing_values'][col] = missing_count
        
        # Check for duplicates
        if 'cid' in df.columns:
            validation_results['duplicate_records'] = df.duplicated(subset=['cid']).sum()
        
        return validation_results
    
    def save_data(self, df: pd.DataFrame, filename: str, format_type: str = 'csv'):
        """
        Save data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            format_type: File format ('csv', 'excel', 'parquet')
        """
        output_path = self.output_dir / filename
        
        if format_type.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format_type.lower() == 'excel':
            df.to_excel(output_path, index=False)
        elif format_type.lower() == 'parquet':
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info(f"Data saved to {output_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Download data from PubChem database")
    parser.add_argument("--compounds", nargs='+', help="Compound names to download")
    parser.add_argument("--cids", nargs='+', type=int, help="PubChem CIDs to download")
    parser.add_argument("--search", type=str, help="Search query for compounds")
    parser.add_argument("--aid", type=int, help="Assay ID for bioassay data")
    parser.add_argument("--max-records", type=int, default=1000,
                       help="Maximum number of records to download")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory")
    parser.add_argument("--format", type=str, default="csv", choices=['csv', 'excel', 'parquet'],
                       help="Output format")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not PUBCHEMPY_AVAILABLE:
        logger.error("PubChemPy not available. Install with: pip install pubchempy")
        sys.exit(1)
    
    # Initialize downloader
    downloader = PubChemDownloader(args.output_dir)
    
    df = pd.DataFrame()
    
    if args.compounds:
        # Download by compound names
        df = downloader.download_compounds_by_name(args.compounds)
        filename = f"pubchem_compounds_by_name.{args.format}"
        
    elif args.cids:
        # Download by CIDs
        df = downloader.download_compounds_by_cid(args.cids)
        filename = f"pubchem_compounds_by_cid.{args.format}"
        
    elif args.search:
        # Search compounds
        df = downloader.search_compounds(args.search, args.max_records)
        filename = f"pubchem_search_{args.search.replace(' ', '_')}.{args.format}"
        
    elif args.aid:
        # Download bioassay data
        df = downloader.download_bioassay_data(args.aid, args.max_records)
        filename = f"pubchem_bioassay_{args.aid}.{args.format}"
        
    else:
        print("Please specify one of: --compounds, --cids, --search, or --aid")
        print("Examples:")
        print("  python download_pubchem.py --compounds aspirin caffeine")
        print("  python download_pubchem.py --cids 2244 2519")
        print("  python download_pubchem.py --search 'kinase inhibitor'")
        print("  python download_pubchem.py --aid 1234")
        sys.exit(1)
    
    if not df.empty:
        # Validate data
        validation_results = downloader.validate_data(df)
        logger.info(f"Validation results: {validation_results}")
        
        # Save data
        downloader.save_data(df, filename, args.format)
        
        print(f"✅ Downloaded {len(df)} records from PubChem")
        print(f"📁 Saved to {downloader.output_dir / filename}")
    else:
        print("❌ No data downloaded")


if __name__ == "__main__":
    main()