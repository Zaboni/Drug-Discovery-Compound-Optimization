#!/usr/bin/env python3
"""
ChEMBL Data Download Script

This script downloads bioactivity data from ChEMBL database.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_CLIENT_AVAILABLE = True
except ImportError:
    CHEMBL_CLIENT_AVAILABLE = False
    # Define dummy client for when not available
    class DummyClient:
        molecule = None
        activity = None
        target = None
        assay = None
    new_client = DummyClient()
    print("ChEMBL webresource client not available. Install with: pip install chembl_webresource_client")

logger = logging.getLogger(__name__)


class ChEMBLDownloader:
    """Downloads data from ChEMBL database."""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize ChEMBL downloader.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        # Use absolute path to ensure correct directory
        if not Path(output_dir).is_absolute():
            # Get the project root directory (parent of scripts directory)
            project_root = Path(__file__).parent.parent
            self.output_dir = project_root / output_dir
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if CHEMBL_CLIENT_AVAILABLE:
            self.molecule = new_client.molecule
            self.activity = new_client.activity
            self.target = new_client.target
            self.assay = new_client.assay
        
    def download_target_activities(self, target_chembl_id: str, 
                                 activity_types: list = None,
                                 max_records: int = 10000) -> pd.DataFrame:
        """
        Download activities for a specific target.
        
        Args:
            target_chembl_id: ChEMBL target ID (e.g., 'CHEMBL279')
            activity_types: List of activity types to download
            max_records: Maximum number of records to download
            
        Returns:
            DataFrame with activity data
        """
        if not CHEMBL_CLIENT_AVAILABLE:
            raise ImportError("ChEMBL webresource client is required")
            
        if activity_types is None:
            activity_types = ['IC50', 'EC50', 'Ki', 'Kd']
        
        logger.info(f"Downloading activities for target {target_chembl_id}")
        
        activities = []
        for activity_type in activity_types:
            try:
                logger.info(f"Downloading {activity_type} activities...")
                
                # Query activities
                activity_data = self.activity.filter(
                    target_chembl_id=target_chembl_id,
                    standard_type=activity_type,
                    standard_value__isnull=False
                ).only([
                    'molecule_chembl_id',
                    'canonical_smiles',
                    'standard_type',
                    'standard_value',
                    'standard_units',
                    'pchembl_value',
                    'assay_chembl_id',
                    'target_chembl_id'
                ])
                
                # Convert to list and limit records
                activity_list = list(activity_data)[:max_records]
                activities.extend(activity_list)
                
                logger.info(f"Downloaded {len(activity_list)} {activity_type} activities")
                time.sleep(1)  # Be nice to the API
                
            except Exception as e:
                logger.error(f"Error downloading {activity_type} activities: {e}")
        
        if activities:
            df = pd.DataFrame(activities)
            logger.info(f"Total activities downloaded: {len(df)}")
            return df
        else:
            logger.warning("No activities downloaded")
            return pd.DataFrame()
    
    def download_compound_data(self, chembl_ids: list) -> pd.DataFrame:
        """
        Download compound data for specific ChEMBL IDs.
        
        Args:
            chembl_ids: List of ChEMBL compound IDs
            
        Returns:
            DataFrame with compound data
        """
        if not CHEMBL_CLIENT_AVAILABLE:
            raise ImportError("ChEMBL webresource client is required")
        
        logger.info(f"Downloading data for {len(chembl_ids)} compounds")
        
        compounds = []
        for chembl_id in tqdm(chembl_ids, desc="Downloading compounds"):
            try:
                compound_data = self.molecule.filter(
                    molecule_chembl_id=chembl_id
                ).only([
                    'molecule_chembl_id',
                    'canonical_smiles',
                    'molecular_weight',
                    'alogp',
                    'psa',
                    'hbd',
                    'hba',
                    'rtb',
                    'aromatic_rings',
                    'heavy_atoms'
                ])
                
                compound_list = list(compound_data)
                compounds.extend(compound_list)
                
            except Exception as e:
                logger.error(f"Error downloading compound {chembl_id}: {e}")
        
        if compounds:
            df = pd.DataFrame(compounds)
            logger.info(f"Downloaded data for {len(df)} compounds")
            return df
        else:
            logger.warning("No compound data downloaded")
            return pd.DataFrame()
    
    def download_assay_data(self, assay_type: str = 'B', max_records: int = 10000) -> pd.DataFrame:
        """
        Download assay data.
        
        Args:
            assay_type: Assay type ('B' for binding, 'F' for functional)
            max_records: Maximum number of records
            
        Returns:
            DataFrame with assay data
        """
        if not CHEMBL_CLIENT_AVAILABLE:
            raise ImportError("ChEMBL webresource client is required")
        
        logger.info(f"Downloading {assay_type} assay data")
        
        try:
            assay_data = self.assay.filter(
                assay_type=assay_type
            ).only([
                'assay_chembl_id',
                'description',
                'assay_type',
                'target_chembl_id',
                'confidence_score'
            ])
            
            assay_list = list(assay_data)[:max_records]
            
            if assay_list:
                df = pd.DataFrame(assay_list)
                logger.info(f"Downloaded {len(df)} assay records")
                return df
            else:
                logger.warning("No assay data downloaded")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error downloading assay data: {e}")
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
                # Define dummy Chem for when RDKit is not available
                class Chem:
                    @staticmethod
                    def MolFromSmiles(smiles):
                        return None
                logger.warning("RDKit not available - skipping SMILES validation")
        
        # Check for missing values in each column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                validation_results['missing_values'][col] = missing_count
        
        # Check for duplicates
        if 'molecule_chembl_id' in df.columns:
            validation_results['duplicate_records'] = df.duplicated(subset=['molecule_chembl_id']).sum()

        return validation_results

    def save_data(self, df: pd.DataFrame, filename: str, format: str = 'csv'):
        """
        Save data to file.

        Args:
            df: DataFrame to save
            filename: Output filename
            format: File format ('csv', 'excel', 'parquet')
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / filename

        # Convert to absolute string path for Windows compatibility
        output_path_str = str(output_path.absolute())

        try:
            # Debug logging
            logger.info(f"Attempting to save to: {output_path_str}")
            logger.info(f"Directory exists: {self.output_dir.exists()}")
            logger.info(f"Directory is dir: {self.output_dir.is_dir()}")

            if format.lower() == 'csv':
                # Create file first to avoid Windows pandas issue
                with open(output_path_str, 'w', encoding='utf-8', newline='') as f:
                    df.to_csv(f, index=False, lineterminator='\n')
            elif format.lower() == 'excel':
                df.to_excel(output_path_str, index=False)
            elif format.lower() == 'parquet':
                df.to_parquet(output_path_str, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {e}")
            logger.error(f"Output directory: {self.output_dir}")
            logger.error(f"Output directory absolute: {self.output_dir.absolute()}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Download data from ChEMBL database")
    parser.add_argument("--target", type=str, help="ChEMBL target ID (e.g., CHEMBL279)")
    parser.add_argument("--activity-types", nargs='+', default=['IC50', 'EC50', 'Ki', 'Kd'],
                       help="Activity types to download")
    parser.add_argument("--max-records", type=int, default=10000,
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
    
    if not CHEMBL_CLIENT_AVAILABLE:
        logger.error("ChEMBL webresource client not available. Install with: pip install chembl_webresource_client")
        sys.exit(1)
    
    # Initialize downloader
    downloader = ChEMBLDownloader(args.output_dir)
    
    if args.target:
        # Download target activities
        df = downloader.download_target_activities(
            args.target, 
            args.activity_types, 
            args.max_records
        )
        
        if not df.empty:
            # Validate data
            validation_results = downloader.validate_data(df)
            logger.info(f"Validation results: {validation_results}")
            
            # Save data
            filename = f"chembl_{args.target}_activities.{args.format}"
            downloader.save_data(df, filename, args.format)
            
            print(f"✅ Downloaded {len(df)} activities for target {args.target}")
            print(f"📁 Saved to {downloader.output_dir / filename}")
        else:
            print(f"❌ No data downloaded for target {args.target}")
    else:
        print("Please specify a target ID with --target")
        print("Example: python download_chembl.py --target CHEMBL279")


if __name__ == "__main__":
    main()