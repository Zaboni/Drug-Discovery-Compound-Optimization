#!/usr/bin/env python3
"""
Tox21 Dataset Download Script

This script downloads toxicity datasets from the Tox21 challenge.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import requests
import io

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

logger = logging.getLogger(__name__)


class Tox21Downloader:
    """Downloads Tox21 toxicity datasets."""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize Tox21 downloader.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tox21 dataset URLs
        self.dataset_urls = {
            'tox21_train': 'https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_data_allsdf',
            'tox21_test': 'https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_testsdf',
            'tox21_score': 'https://tripod.nih.gov/tox21/challenge/download?id=tox21_10k_challenge_scoresdf'
        }
        
        # Alternative URLs (DeepChem repository)
        self.deepchem_urls = {
            'tox21_train': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
            'tox21_full': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21_full.csv.gz'
        }
        
        # Assay information
        self.assays = {
            'NR-AR': 'Androgen Receptor',
            'NR-AR-LBD': 'Androgen Receptor Ligand Binding Domain',
            'NR-AhR': 'Aryl Hydrocarbon Receptor',
            'NR-Aromatase': 'Aromatase',
            'NR-ER': 'Estrogen Receptor Alpha',
            'NR-ER-LBD': 'Estrogen Receptor Alpha Ligand Binding Domain',
            'NR-PPAR-gamma': 'Peroxisome Proliferator-Activated Receptor Gamma',
            'SR-ARE': 'Antioxidant Response Element',
            'SR-ATAD5': 'ATAD5',
            'SR-HSE': 'Heat Shock Factor Response Element',
            'SR-MMP': 'Mitochondrial Membrane Potential',
            'SR-p53': 'p53'
        }
    
    def download_from_deepchem(self, dataset_name: str = 'tox21_train') -> pd.DataFrame:
        """
        Download Tox21 dataset from DeepChem repository.
        
        Args:
            dataset_name: Name of dataset to download
            
        Returns:
            DataFrame with Tox21 data
        """
        if dataset_name not in self.deepchem_urls:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        url = self.deepchem_urls[dataset_name]
        logger.info(f"Downloading {dataset_name} from DeepChem repository")
        
        try:
            # Download and read compressed CSV
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            # Read compressed data
            df = pd.read_csv(io.BytesIO(response.content), compression='gzip')
            
            logger.info(f"Downloaded {len(df)} records from {dataset_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return pd.DataFrame()
    
    def download_from_nih(self, dataset_name: str = 'tox21_train') -> pd.DataFrame:
        """
        Download Tox21 dataset from NIH repository.
        
        Args:
            dataset_name: Name of dataset to download
            
        Returns:
            DataFrame with Tox21 data
        """
        if dataset_name not in self.dataset_urls:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        url = self.dataset_urls[dataset_name]
        logger.info(f"Downloading {dataset_name} from NIH repository")
        
        try:
            # Download SDF file
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Save SDF file
            sdf_file = self.output_dir / f"{dataset_name}.sdf"
            with open(sdf_file, 'wb') as f:
                f.write(response.content)
            
            # Convert SDF to DataFrame using RDKit
            try:
                from rdkit.Chem import PandasTools
                df = PandasTools.LoadSDF(str(sdf_file))
                logger.info(f"Loaded {len(df)} records from SDF file")
                return df
            except ImportError:
                PandasTools = None
                logger.error("RDKit not available - cannot process SDF file")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error downloading {dataset_name}: {e}")
            return pd.DataFrame()
    
    def download_all_datasets(self, source: str = 'deepchem') -> dict:
        """
        Download all available Tox21 datasets.
        
        Args:
            source: Data source ('deepchem' or 'nih')
            
        Returns:
            Dictionary of DataFrames
        """
        datasets = {}
        
        if source == 'deepchem':
            for dataset_name in self.deepchem_urls.keys():
                logger.info(f"Downloading {dataset_name}...")
                df = self.download_from_deepchem(dataset_name)
                if not df.empty:
                    datasets[dataset_name] = df
        
        elif source == 'nih':
            for dataset_name in self.dataset_urls.keys():
                logger.info(f"Downloading {dataset_name}...")
                df = self.download_from_nih(dataset_name)
                if not df.empty:
                    datasets[dataset_name] = df
        
        else:
            raise ValueError("Source must be 'deepchem' or 'nih'")
        
        return datasets
    
    def process_tox21_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Tox21 data for machine learning.
        
        Args:
            df: Raw Tox21 DataFrame
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        
        # Identify assay columns
        assay_columns = [col for col in df.columns if any(assay in col for assay in self.assays.keys())]
        
        # Convert assay results to binary (0/1) and handle missing values
        for col in assay_columns:
            if col in df_processed.columns:
                # Convert to numeric, replacing non-numeric with NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Convert to binary (assuming positive values are active)
                df_processed[col] = (df_processed[col] > 0).astype(int)
        
        # Add metadata
        df_processed['num_assays_tested'] = df_processed[assay_columns].notna().sum(axis=1)
        df_processed['num_active_assays'] = df_processed[assay_columns].sum(axis=1)
        df_processed['activity_ratio'] = df_processed['num_active_assays'] / df_processed['num_assays_tested']
        
        logger.info(f"Processed {len(df_processed)} compounds with {len(assay_columns)} assays")
        return df_processed
    
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Validate Tox21 data.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_records': len(df),
            'missing_smiles': 0,
            'invalid_smiles': 0,
            'assay_coverage': {},
            'missing_values': {},
            'duplicate_records': 0
        }
        
        # Check SMILES
        smiles_columns = [col for col in df.columns if 'smiles' in col.lower()]
        if smiles_columns:
            smiles_col = smiles_columns[0]
            validation_results['missing_smiles'] = df[smiles_col].isnull().sum()
            
            # Check for invalid SMILES
            try:
                from rdkit import Chem
                invalid_count = 0
                for smiles in df[smiles_col].dropna():
                    mol = Chem.MolFromSmiles(str(smiles))
                    if mol is None:
                        invalid_count += 1
                validation_results['invalid_smiles'] = invalid_count
            except ImportError:
                Chem = None
                logger.warning("RDKit not available - skipping SMILES validation")
        
        # Check assay coverage
        assay_columns = [col for col in df.columns if any(assay in col for assay in self.assays.keys())]
        for col in assay_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                validation_results['assay_coverage'][col] = {
                    'tested': non_null_count,
                    'coverage': non_null_count / len(df) * 100
                }
        
        # Check for missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                validation_results['missing_values'][col] = missing_count
        
        # Check for duplicates
        if smiles_columns:
            validation_results['duplicate_records'] = df.duplicated(subset=[smiles_columns[0]]).sum()
        
        return validation_results
    
    def create_assay_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create summary statistics for each assay.
        
        Args:
            df: Tox21 DataFrame
            
        Returns:
            DataFrame with assay summary
        """
        assay_columns = [col for col in df.columns if any(assay in col for assay in self.assays.keys())]
        
        summary_data = []
        for col in assay_columns:
            if col in df.columns:
                assay_data = df[col].dropna()
                
                summary = {
                    'assay': col,
                    'description': self.assays.get(col.split('_')[0], 'Unknown'),
                    'total_tested': len(assay_data),
                    'num_active': assay_data.sum(),
                    'num_inactive': len(assay_data) - assay_data.sum(),
                    'activity_rate': assay_data.mean() * 100,
                    'coverage': len(assay_data) / len(df) * 100
                }
                summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
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
    parser = argparse.ArgumentParser(description="Download Tox21 toxicity datasets")
    parser.add_argument("--dataset", type=str, choices=['tox21_train', 'tox21_test', 'tox21_score', 'tox21_full', 'all'],
                       default='tox21_train', help="Dataset to download")
    parser.add_argument("--source", type=str, choices=['deepchem', 'nih'], default='deepchem',
                       help="Data source")
    parser.add_argument("--process", action='store_true', help="Process data for ML")
    parser.add_argument("--summary", action='store_true', help="Create assay summary")
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
    
    # Initialize downloader
    downloader = Tox21Downloader(args.output_dir)
    
    if args.dataset == 'all':
        # Download all datasets
        datasets = downloader.download_all_datasets(args.source)
        
        for dataset_name, df in datasets.items():
            if not df.empty:
                # Process data if requested
                if args.process:
                    df = downloader.process_tox21_data(df)
                
                # Validate data
                validation_results = downloader.validate_data(df)
                logger.info(f"Validation results for {dataset_name}: {validation_results}")
                
                # Create assay summary if requested
                if args.summary:
                    summary_df = downloader.create_assay_summary(df)
                    summary_filename = f"tox21_{dataset_name}_assay_summary.{args.format}"
                    downloader.save_data(summary_df, summary_filename, args.format)
                
                # Save data
                filename = f"tox21_{dataset_name}.{args.format}"
                downloader.save_data(df, filename, args.format)
                
                print(f"✅ Downloaded and saved {dataset_name}: {len(df)} records")
    
    else:
        # Download specific dataset
        if args.source == 'deepchem':
            df = downloader.download_from_deepchem(args.dataset)
        else:
            df = downloader.download_from_nih(args.dataset)
        
        if not df.empty:
            # Process data if requested
            if args.process:
                df = downloader.process_tox21_data(df)
            
            # Validate data
            validation_results = downloader.validate_data(df)
            logger.info(f"Validation results: {validation_results}")
            
            # Create assay summary if requested
            if args.summary:
                summary_df = downloader.create_assay_summary(df)
                summary_filename = f"tox21_{args.dataset}_assay_summary.{args.format}"
                downloader.save_data(summary_df, summary_filename, args.format)
                print(f"📊 Created assay summary: {len(summary_df)} assays")
            
            # Save data
            filename = f"tox21_{args.dataset}.{args.format}"
            downloader.save_data(df, filename, args.format)
            
            print(f"✅ Downloaded {args.dataset}: {len(df)} records")
            print(f"📁 Saved to {downloader.output_dir / filename}")
        else:
            print(f"❌ No data downloaded for {args.dataset}")


if __name__ == "__main__":
    main()