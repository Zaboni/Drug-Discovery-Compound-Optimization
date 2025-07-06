#!/usr/bin/env python3
"""
Data Processing Pipeline Script

This script provides a complete command-line interface for processing molecular datasets
from raw data to model-ready features.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from data_processing import (
        MolecularDataLoader, MolecularPreprocessor,
        FeatureEnginerator, DataSplitter
    )
    from utils import assess_data_quality, generate_data_report
    from logging_config import setup_logging, get_logger
    DATA_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data processing modules: {e}")
    DATA_PROCESSING_AVAILABLE = False

    # Define dummy classes for when modules are not available
    class MolecularDataLoader:
        def auto_load(self, path, col): return pd.DataFrame()

    class MolecularPreprocessor:
        def __init__(self, config=None): pass
        def validate_molecules(self, df, col='smiles'): return df
        def standardize_molecules(self, df, col='smiles'): return df
        def remove_duplicates(self, df, col): return df
        def apply_quality_filters(self, df): return df

    class FeatureEnginerator:
        def __init__(self, config=None): pass
        def extract_molecular_descriptors(self, df, col): return df
        def extract_molecular_fingerprints(self, df, col): return df

    class DataSplitter:
        def __init__(self, config=None): pass
        def random_split(self, df, *args, **kwargs): return df, df, df
        def scaffold_split(self, df, *args, **kwargs): return df, df, df
        def cluster_split(self, df, *args, **kwargs): return df, df, df

    def assess_data_quality(df, col): return {}
    def generate_data_report(df, col, target=None): return ""
    def setup_logging(log_level="INFO"): pass
    def get_logger(name):
        import logging
        return logging.getLogger(name)


class DataProcessingPipeline:
    """Complete data processing pipeline for molecular datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processing pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.loader = MolecularDataLoader()
        self.preprocessor = MolecularPreprocessor(config)
        self.feature_eng = FeatureEnginerator(config)
        self.splitter = DataSplitter(config)
        
        # Pipeline state
        self.raw_data = None
        self.processed_data = None
        self.features_data = None
        self.splits = None
        
    def load_data(self, input_path: str, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Load molecular data from file.
        
        Args:
            input_path: Path to input data file
            smiles_column: Name of SMILES column
            
        Returns:
            Loaded DataFrame
        """
        self.logger.info(f"Loading data from {input_path}")
        
        try:
            self.raw_data = self.loader.auto_load(input_path, smiles_column)
            self.logger.info(f"Loaded {len(self.raw_data)} records")
            
            # Log basic info about the dataset
            self.logger.info(f"Columns: {list(self.raw_data.columns)}")
            self.logger.info(f"Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return self.raw_data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def assess_quality(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> Dict[str, Any]:
        """
        Assess data quality.
        
        Args:
            df: DataFrame to assess
            smiles_column: Name of SMILES column
            
        Returns:
            Quality assessment results
        """
        self.logger.info("Assessing data quality")
        
        try:
            quality_report = assess_data_quality(df, smiles_column)
            
            # Log key quality metrics
            self.logger.info(f"Overall quality score: {quality_report.get('overall_quality_score', 'N/A'):.1f}/100")
            
            # Log missing data
            missing_data = quality_report.get('missing_data', {})
            for col, info in missing_data.items():
                if info['count'] > 0:
                    self.logger.warning(f"Missing data in {col}: {info['count']} ({info['percentage']:.1f}%)")
            
            # Log SMILES quality
            smiles_quality = quality_report.get('smiles_quality', {})
            if 'validity_rate' in smiles_quality:
                self.logger.info(f"SMILES validity rate: {smiles_quality['validity_rate']:.1f}%")
            
            # Log recommendations
            recommendations = quality_report.get('recommendations', [])
            if recommendations:
                self.logger.info("Data quality recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    self.logger.info(f"  {i}. {rec}")
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Error assessing data quality: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame, smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Preprocess molecular data.
        
        Args:
            df: Input DataFrame
            smiles_column: Name of SMILES column
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Preprocessing molecular data")
        
        try:
            # Validate molecules
            self.logger.info("Validating SMILES strings")
            df_processed = self.preprocessor.validate_molecules(df, smiles_column)
            
            valid_count = df_processed.get('valid', pd.Series([True]*len(df))).sum()
            self.logger.info(f"Valid molecules: {valid_count}/{len(df_processed)}")
            
            # Standardize molecules
            self.logger.info("Standardizing molecules")
            df_processed = self.preprocessor.standardize_molecules(df_processed, smiles_column)
            
            # Remove duplicates
            if self.config.get('preprocessing', {}).get('remove_duplicates', True):
                self.logger.info("Removing duplicates")
                initial_count = len(df_processed)
                df_processed = self.preprocessor.remove_duplicates(
                    df_processed, 
                    'canonical_smiles' if 'canonical_smiles' in df_processed.columns else smiles_column
                )
                removed_count = initial_count - len(df_processed)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} duplicate molecules")
            
            # Apply quality filters
            if self.config.get('preprocessing', {}).get('apply_filters', True):
                self.logger.info("Applying quality filters")
                initial_count = len(df_processed)
                df_processed = self.preprocessor.apply_quality_filters(df_processed)
                removed_count = initial_count - len(df_processed)
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} molecules by quality filters")
            
            self.processed_data = df_processed
            self.logger.info(f"Preprocessing completed: {len(df_processed)} molecules remaining")
            
            return df_processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {e}")
            raise
    
    def extract_features(self, df: pd.DataFrame, smiles_column: str = 'canonical_smiles') -> pd.DataFrame:
        """
        Extract molecular features.
        
        Args:
            df: Input DataFrame
            smiles_column: Name of SMILES column
            
        Returns:
            DataFrame with extracted features
        """
        self.logger.info("Extracting molecular features")
        
        try:
            # Extract molecular descriptors
            self.logger.info("Calculating molecular descriptors")
            df_features = self.feature_eng.extract_molecular_descriptors(df, smiles_column)
            
            # Extract fingerprints if configured
            if self.config.get('feature_extraction', {}).get('fingerprints', {}).get('enabled', True):
                self.logger.info("Calculating molecular fingerprints")
                df_features = self.feature_eng.extract_molecular_fingerprints(df_features, smiles_column)
            
            # Log feature extraction results
            new_features = [col for col in df_features.columns if col not in df.columns]
            self.logger.info(f"Extracted {len(new_features)} new features")
            
            self.features_data = df_features
            return df_features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame, target_column: Optional[str] = None,
                  split_method: str = 'random') -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: Input DataFrame
            target_column: Target column for stratification
            split_method: Splitting method ('random', 'scaffold', 'cluster')
            
        Returns:
            Dictionary with train/val/test DataFrames
        """
        self.logger.info(f"Splitting data using {split_method} method")
        
        try:
            # Get split ratios from config
            split_config = self.config.get('data_splitting', {}).get(split_method, {})
            train_ratio = split_config.get('train_ratio', 0.7)
            val_ratio = split_config.get('val_ratio', 0.15)
            test_ratio = split_config.get('test_ratio', 0.15)
            
            if split_method == 'random':
                stratify = split_config.get('stratify', False) and target_column is not None
                train_df, val_df, test_df = self.splitter.random_split(
                    df, target_column, train_ratio, val_ratio, test_ratio, stratify
                )
            
            elif split_method == 'scaffold':
                smiles_col = 'canonical_smiles' if 'canonical_smiles' in df.columns else 'smiles'
                train_df, val_df, test_df = self.splitter.scaffold_split(
                    df, smiles_col, train_ratio, val_ratio, test_ratio
                )
            
            elif split_method == 'cluster':
                # Use numeric features for clustering
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols if col not in ['activity', target_column]]
                
                if len(feature_cols) < 2:
                    self.logger.warning("Insufficient features for cluster split, using random split")
                    train_df, val_df, test_df = self.splitter.random_split(
                        df, target_column, train_ratio, val_ratio, test_ratio
                    )
                else:
                    train_df, val_df, test_df = self.splitter.cluster_split(
                        df, feature_cols, train_ratio, val_ratio, test_ratio
                    )
            
            else:
                raise ValueError(f"Unknown split method: {split_method}")
            
            # Log split results
            self.logger.info(f"Data split completed:")
            self.logger.info(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
            self.logger.info(f"  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
            self.logger.info(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
            
            splits = {
                'train': train_df,
                'validation': val_df,
                'test': test_df
            }
            
            self.splits = splits
            return splits
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {e}")
            raise
    
    def save_results(self, output_dir: str, save_splits: bool = True, 
                    save_reports: bool = True) -> Dict[str, str]:
        """
        Save processing results.
        
        Args:
            output_dir: Output directory
            save_splits: Whether to save data splits
            save_reports: Whether to save quality reports
            
        Returns:
            Dictionary of saved file paths
        """
        self.logger.info(f"Saving results to {output_dir}")
        
        # Simplify directory handling for Windows compatibility
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = output_dir.resolve()
            self.logger.info(f"Using output directory: {output_dir}")
        except Exception as e:
            self.logger.warning(f"Could not create/access output directory: {e}")
            output_dir = Path('.').resolve()
            self.logger.warning(f"Using current directory as fallback: {output_dir}")

        saved_files = {}
        
        try:
            # Save processed data
            if self.features_data is not None:
                processed_file = output_dir / 'processed_data.csv'
                try:
                    self.features_data.to_csv(processed_file, index=False)
                    saved_files['processed_data'] = str(processed_file)
                    self.logger.info(f"Saved processed data to {processed_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save processed data: {e}")

            # Save data splits
            if save_splits and self.splits is not None:
                for split_name, split_df in self.splits.items():
                    split_file = output_dir / f'{split_name}_data.csv'
                    try:
                        split_df.to_csv(split_file, index=False)
                        saved_files[f'{split_name}_data'] = str(split_file)
                        self.logger.info(f"Saved {split_name} data to {split_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to save {split_name} data: {e}")

            # Save quality report
            if save_reports and self.processed_data is not None:
                try:
                    report_text = generate_data_report(
                        self.processed_data,
                        'canonical_smiles' if 'canonical_smiles' in self.processed_data.columns else 'smiles'
                    )
                    report_file = output_dir / 'data_quality_report.txt'
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(report_text)
                    saved_files['quality_report'] = str(report_file)
                    self.logger.info(f"Saved quality report to {report_file}")
                except Exception as e:
                    self.logger.error(f"Failed to save quality report: {e}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
            raise
    
    def run_pipeline(self, input_path: str, output_dir: str, 
                    smiles_column: str = 'smiles', target_column: Optional[str] = None,
                    split_method: str = 'random') -> Dict[str, str]:
        """
        Run the complete data processing pipeline.
        
        Args:
            input_path: Path to input data file
            output_dir: Output directory for results
            smiles_column: Name of SMILES column
            target_column: Target column for splitting
            split_method: Data splitting method
            
        Returns:
            Dictionary of saved file paths
        """
        self.logger.info("Starting data processing pipeline")
        
        try:
            # Step 1: Load data
            df = self.load_data(input_path, smiles_column)
            
            # Step 2: Assess quality
            quality_report = self.assess_quality(df, smiles_column)
            
            # Step 3: Preprocess data
            df_processed = self.preprocess_data(df, smiles_column)
            
            # Step 4: Extract features
            df_features = self.extract_features(df_processed)
            
            # Step 5: Split data
            splits = self.split_data(df_features, target_column, split_method)
            
            # Step 6: Save results
            saved_files = self.save_results(output_dir)
            
            self.logger.info("Data processing pipeline completed successfully")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Complete data processing pipeline for molecular datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python process_data.py input.csv --output-dir processed/

  # With custom SMILES column and target
  python process_data.py data.csv --smiles-column compound_smiles --target-column activity

  # Using scaffold splitting
  python process_data.py molecules.csv --split-method scaffold --output-dir results/

  # Skip feature extraction
  python process_data.py data.csv --no-features --output-dir simple/
        """
    )
    
    # Input/Output arguments
    parser.add_argument("input_file", help="Input data file (CSV, Excel, SDF, or SMILES)")
    parser.add_argument("--output-dir", "-o", default="data/processed", 
                       help="Output directory (default: data/processed)")
    
    # Data columns
    parser.add_argument("--smiles-column", default="smiles", 
                       help="Name of SMILES column (default: smiles)")
    parser.add_argument("--target-column", 
                       help="Target column for stratified splitting")
    
    # Processing options
    parser.add_argument("--split-method", choices=['random', 'scaffold', 'cluster'], 
                       default='random', help="Data splitting method (default: random)")
    parser.add_argument("--no-features", action='store_true', 
                       help="Skip feature extraction")
    parser.add_argument("--no-splits", action='store_true', 
                       help="Skip data splitting")
    parser.add_argument("--no-reports", action='store_true', 
                       help="Skip quality reports")
    
    # Configuration
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Setup logging with basic configuration if setup_logging fails
    try:
        setup_logging(log_level=args.log_level)
    except Exception as e:
        # Fallback to basic logging if setup fails
        import logging
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        print(f"Warning: Advanced logging setup failed ({e}), using basic logging")

    logger = get_logger(__name__)
    
    if not DATA_PROCESSING_AVAILABLE:
        logger.error("Data processing modules not available")
        sys.exit(1)
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        try:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.warning(f"Could not load configuration: {e}")
    
    # Check input file
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = DataProcessingPipeline(config)
        
        # Run pipeline steps
        logger.info(f"Processing {args.input_file}")
        
        # Load data
        df = pipeline.load_data(args.input_file, args.smiles_column)
        
        # Assess quality
        quality_report = pipeline.assess_quality(df, args.smiles_column)
        
        # Preprocess
        df_processed = pipeline.preprocess_data(df, args.smiles_column)
        
        # Extract features (optional)
        if not args.no_features:
            df_features = pipeline.extract_features(df_processed)
        else:
            df_features = df_processed
            logger.info("Skipping feature extraction")
        
        # Split data (optional)
        if not args.no_splits:
            splits = pipeline.split_data(df_features, args.target_column, args.split_method)
        else:
            logger.info("Skipping data splitting")
        
        # Save results
        saved_files = pipeline.save_results(
            args.output_dir, 
            save_splits=not args.no_splits,
            save_reports=not args.no_reports
        )
        
        # Print summary
        print("\n" + "="*60)
        print("DATA PROCESSING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Input file: {args.input_file}")
        print(f"Output directory: {args.output_dir}")
        print(f"Original records: {len(df)}")
        print(f"Processed records: {len(df_features)}")
        
        if 'overall_quality_score' in quality_report:
            print(f"Quality score: {quality_report['overall_quality_score']:.1f}/100")
        
        print(f"\nSaved files:")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {file_path}")
        
        print(f"\n✅ Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()