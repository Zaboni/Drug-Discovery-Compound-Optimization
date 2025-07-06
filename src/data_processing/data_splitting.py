"""
DataSplitter for train/validation/test splits.

This module handles various data splitting strategies including random,
scaffold-based, and cluster-based splitting for molecular datasets.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .core import (
    RDKIT_AVAILABLE, SKLEARN_AVAILABLE, Chem, MurckoScaffold,
    train_test_split, KMeans, logger
)


class DataSplitter:
    """
    Handles train/validation/test splits with stratification and scaffold-based splitting.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data splitter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def random_split(self, df: pd.DataFrame, target_column: Optional[str] = None,
                    train_ratio: float = 0.7, val_ratio: float = 0.15,
                    test_ratio: float = 0.15, stratify: bool = False,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform random train/validation/test split.

        Args:
            df: DataFrame to split
            target_column: Target column for stratification
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify: Whether to stratify split
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for data splitting")

        # Normalize ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

        # First split: separate test set
        if stratify and target_column and target_column in df.columns:
            train_val_df, test_df = train_test_split(
                df, test_size=test_ratio, stratify=df[target_column],
                random_state=random_state
            )
        else:
            train_val_df, test_df = train_test_split(
                df, test_size=test_ratio, random_state=random_state
            )

        # Second split: separate train and validation
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)

        if stratify and target_column and target_column in train_val_df.columns:
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_ratio_adjusted,
                stratify=train_val_df[target_column], random_state=random_state
            )
        else:
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_ratio_adjusted, random_state=random_state
            )

        logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def scaffold_split(self, df: pd.DataFrame, smiles_column: str = 'canonical_smiles',
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform scaffold-based split for better generalization.

        Args:
            df: DataFrame to split
            smiles_column: SMILES column name
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for scaffold splitting")

        # Calculate scaffolds
        scaffolds = {}
        for idx, smiles in enumerate(df[smiles_column]):
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                    if scaffold not in scaffolds:
                        scaffolds[scaffold] = []
                    scaffolds[scaffold].append(idx)
            except Exception as e:
                logger.warning(f"Error calculating scaffold for {smiles}: {e}")
                # Assign to a unique scaffold
                unique_scaffold = f"unique_{idx}"
                scaffolds[unique_scaffold] = [idx]

        # Sort scaffolds by size (largest first)
        scaffold_sets = list(scaffolds.values())
        scaffold_sets.sort(key=len, reverse=True)

        # Assign scaffolds to splits
        train_indices, val_indices, test_indices = [], [], []
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)

        for scaffold_set in scaffold_sets:
            if len(train_indices) < train_size:
                train_indices.extend(scaffold_set)
            elif len(val_indices) < val_size:
                val_indices.extend(scaffold_set)
            else:
                test_indices.extend(scaffold_set)

        train_df = df.iloc[train_indices].reset_index(drop=True)
        val_df = df.iloc[val_indices].reset_index(drop=True)
        test_df = df.iloc[test_indices].reset_index(drop=True)

        logger.info(f"Scaffold split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def cluster_split(self, df: pd.DataFrame, features: List[str],
                     train_ratio: float = 0.7, val_ratio: float = 0.15,
                     test_ratio: float = 0.15, n_clusters: Optional[int] = None,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform cluster-based split.

        Args:
            df: DataFrame to split
            features: Feature columns for clustering
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            n_clusters: Number of clusters (auto if None)
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for cluster splitting")

        # Prepare feature matrix
        feature_matrix = df[features].fillna(0).values

        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(50, len(df) // 10)  # Heuristic

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(feature_matrix)

        # Create cluster-based splits
        cluster_df = df.copy()
        cluster_df['cluster'] = clusters

        # Split clusters
        unique_clusters = np.unique(clusters)
        np.random.seed(random_state)
        np.random.shuffle(unique_clusters)

        n_train_clusters = int(len(unique_clusters) * train_ratio)
        n_val_clusters = int(len(unique_clusters) * val_ratio)

        train_clusters = unique_clusters[:n_train_clusters]
        val_clusters = unique_clusters[n_train_clusters:n_train_clusters + n_val_clusters]
        test_clusters = unique_clusters[n_train_clusters + n_val_clusters:]

        train_df = cluster_df[cluster_df['cluster'].isin(train_clusters)].drop('cluster', axis=1)
        val_df = cluster_df[cluster_df['cluster'].isin(val_clusters)].drop('cluster', axis=1)
        test_df = cluster_df[cluster_df['cluster'].isin(test_clusters)].drop('cluster', axis=1)

        logger.info(f"Cluster split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df

    def get_advanced_strategies(self):
        """Get access to advanced splitting strategies."""
        from .splitting_strategies import AdvancedSplittingStrategies
        return AdvancedSplittingStrategies(self.config)

