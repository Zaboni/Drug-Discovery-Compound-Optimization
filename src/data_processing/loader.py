"""
MolecularDataLoader for loading molecular data from various file formats.

This module handles loading of molecular data from SMILES files, SDF files,
CSV files, and Excel files with automatic format detection.
"""

from typing import Union, Optional
import pandas as pd
from pathlib import Path

from .core import RDKIT_AVAILABLE, PandasTools, logger


class MolecularDataLoader:
    """
    Handles loading molecular data from various file formats including SMILES, SDF, and CSV.
    """

    def __init__(self, config: Optional[dict] = None):
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

    def validate_data_format(self, df: pd.DataFrame, required_columns: list = None) -> bool:
        """
        Validate that loaded data has required format.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid format, False otherwise
        """
        if df.empty:
            logger.warning("DataFrame is empty")
            return False

        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

        logger.info(f"Data format validation passed for DataFrame with shape {df.shape}")
        return True

    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a data file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File not found"}

        try:
            info = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_format": file_path.suffix.lower(),
                "supported": file_path.suffix.lower() in ['.csv', '.xlsx', '.xls', '.sdf', '.smi', '.txt']
            }

            # Try to get basic data info without fully loading
            if info["file_format"] == '.csv':
                try:
                    # Read just the first few rows to get column info
                    sample_df = pd.read_csv(file_path, nrows=5)
                    info["columns"] = list(sample_df.columns)
                    info["sample_records"] = len(sample_df)
                except Exception:
                    info["columns"] = "Could not read columns"

            return info

        except Exception as e:
            return {"error": str(e)}

    def batch_load_files(self, file_paths: list, smiles_column: str = 'smiles') -> dict:
        """
        Load multiple files and combine them.

        Args:
            file_paths: List of file paths to load
            smiles_column: Name of SMILES column

        Returns:
            Dictionary with combined data and loading statistics
        """
        all_dataframes = []
        loading_stats = {
            "total_files": len(file_paths),
            "successful_loads": 0,
            "failed_loads": 0,
            "total_records": 0,
            "errors": []
        }

        for file_path in file_paths:
            try:
                df = self.auto_load(file_path, smiles_column)
                all_dataframes.append(df)
                loading_stats["successful_loads"] += 1
                loading_stats["total_records"] += len(df)
                logger.info(f"Successfully loaded {file_path}")
            except Exception as e:
                loading_stats["failed_loads"] += 1
                loading_stats["errors"].append(f"{file_path}: {str(e)}")
                logger.error(f"Failed to load {file_path}: {e}")

        # Combine all dataframes if any were loaded successfully
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            logger.info(f"Combined {len(all_dataframes)} files into dataset with {len(combined_df)} records")
        else:
            combined_df = pd.DataFrame()
            logger.warning("No files were loaded successfully")

        return {
            "data": combined_df,
            "stats": loading_stats
        }