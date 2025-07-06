"""
Advanced splitting strategies for molecular data.

This module contains specialized splitting methods including temporal,
balanced, and validation utilities for data splits.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path

from .core import logger


class AdvancedSplittingStrategies:
    """Advanced splitting strategies for molecular datasets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced splitting strategies."""
        self.config = config or {}

    def temporal_split(self, df: pd.DataFrame, date_column: str,
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform temporal split based on date column.

        Args:
            df: DataFrame to split
            date_column: Name of date column
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame")

        # Sort by date
        df_sorted = df.sort_values(date_column).reset_index(drop=True)

        # Calculate split indices
        total_size = len(df_sorted)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        # Split data
        train_df = df_sorted.iloc[:train_size].reset_index(drop=True)
        val_df = df_sorted.iloc[train_size:train_size + val_size].reset_index(drop=True)
        test_df = df_sorted.iloc[train_size + val_size:].reset_index(drop=True)

        logger.info(f"Temporal split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def balanced_split(self, df: pd.DataFrame, target_column: str,
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      test_ratio: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform balanced split ensuring equal representation of target classes.

        Args:
            df: DataFrame to split
            target_column: Target column for balancing
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")

        from .data_splitting import DataSplitter
        splitter = DataSplitter()

        # Get unique classes
        unique_classes = df[target_column].value_counts()
        
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for class_value in unique_classes.index:
            # Get data for this class
            class_df = df[df[target_column] == class_value].reset_index(drop=True)
            
            # Split this class
            class_train, class_val, class_test = splitter.random_split(
                class_df, target_column=None,
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                random_state=random_state
            )
            
            train_dfs.append(class_train)
            val_dfs.append(class_val)
            test_dfs.append(class_test)

        # Combine all classes
        train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)

        logger.info(f"Balanced split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def get_split_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Calculate statistics for data splits.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            target_column: Target column for class distribution analysis

        Returns:
            Dictionary with split statistics
        """
        total_size = len(train_df) + len(val_df) + len(test_df)
        
        stats = {
            'total_samples': total_size,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'train_ratio': len(train_df) / total_size,
            'val_ratio': len(val_df) / total_size,
            'test_ratio': len(test_df) / total_size
        }

        # Add target distribution if target column is provided
        if target_column and target_column in train_df.columns:
            stats['target_distribution'] = {
                'train': dict(train_df[target_column].value_counts()),
                'val': dict(val_df[target_column].value_counts()),
                'test': dict(test_df[target_column].value_counts())
            }

        return stats

    def validate_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that splits are correct.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            original_df: Original DataFrame before splitting

        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # Check total size
        total_split_size = len(train_df) + len(val_df) + len(test_df)
        validation['size_preserved'] = total_split_size == len(original_df)
        
        # Check no overlap between splits
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        validation['no_train_val_overlap'] = len(train_indices & val_indices) == 0
        validation['no_train_test_overlap'] = len(train_indices & test_indices) == 0
        validation['no_val_test_overlap'] = len(val_indices & test_indices) == 0
        
        # Check all indices are covered
        all_split_indices = train_indices | val_indices | test_indices
        original_indices = set(original_df.index)
        validation['all_indices_covered'] = all_split_indices == original_indices
        
        # Overall validation
        validation['valid_split'] = all(validation.values())
        
        return validation

    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   test_df: pd.DataFrame, output_dir: str, prefix: str = "") -> Dict[str, str]:
        """
        Save data splits to files.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Output directory
            prefix: Prefix for filenames

        Returns:
            Dictionary with saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save each split
        splits = {'train': train_df, 'val': val_df, 'test': test_df}
        
        for split_name, split_df in splits.items():
            filename = f"{prefix}{split_name}_data.csv" if prefix else f"{split_name}_data.csv"
            file_path = output_path / filename
            split_df.to_csv(file_path, index=False)
            saved_files[split_name] = str(file_path)
            logger.info(f"Saved {split_name} split to {file_path}")
        
        return saved_files