"""
Utility Functions for Drug Discovery Compound Optimization

This module contains various utility functions for molecular processing,
visualization, model interpretation, and general helper functions.
"""

import logging
import os
import random
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    # Define dummy classes for when RDKit is not available
    class Chem:
        @staticmethod
        def MolFromSmiles(smiles): return None
        @staticmethod
        def MolToSmiles(mol): return ""
    class Descriptors:
        @staticmethod
        def MolWt(mol): return 0.0
        @staticmethod
        def MolLogP(mol): return 0.0
        @staticmethod
        def TPSA(mol): return 0.0
        @staticmethod
        def NumRotatableBonds(mol): return 0
        @staticmethod
        def NumHDonors(mol): return 0
        @staticmethod
        def NumHAcceptors(mol): return 0
        @staticmethod
        def NumAromaticRings(mol): return 0
        @staticmethod
        def RingCount(mol): return 0
        @staticmethod
        def MolMR(mol): return 0.0
    class DataStructs:
        @staticmethod
        def TanimotoSimilarity(fp1, fp2): return 0.0
        @staticmethod
        def DiceSimilarity(fp1, fp2): return 0.0
        @staticmethod
        def CosineSimilarity(fp1, fp2): return 0.0
    def GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048): return None
    logging.warning("RDKit not available. Some utility functions will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    # Define dummy classes for when matplotlib is not available
    class plt:
        @staticmethod
        def subplots(*args, **kwargs): return None, None
        @staticmethod
        def figure(*args, **kwargs): return None
        @staticmethod
        def show(): pass
        @staticmethod
        def savefig(*args, **kwargs): pass
        @staticmethod
        def tight_layout(): pass
    class sns:
        @staticmethod
        def histplot(*args, **kwargs): pass
        @staticmethod
        def heatmap(*args, **kwargs): pass
    logging.warning("Matplotlib/Seaborn not available. Plotting functions will be limited.")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy torch for when not available
    class torch:
        @staticmethod
        def manual_seed(seed): pass
        class cuda:
            @staticmethod
            def is_available(): return False
            @staticmethod
            def manual_seed(seed): pass
            @staticmethod
            def manual_seed_all(seed): pass
        class backends:
            class cudnn:
                deterministic = True
                benchmark = False
    logging.warning("PyTorch not available. Some utility functions will be limited.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Model interpretation functions will be limited.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Define dummy tqdm for when not available
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if iterable is not None:
            return iterable
        else:
            class DummyTqdm:
                def __init__(self, total, desc):
                    self.total = total
                    self.desc = desc
                def update(self, n=1): pass
                def close(self): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyTqdm(total, desc)

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def validate_smiles(smiles: str) -> bool:
    """
    Validate a SMILES string.
    
    Args:
        smiles: SMILES string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available - cannot validate SMILES")
        return False
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize a SMILES string.
    
    Args:
        smiles: Input SMILES string
        
    Returns:
        Canonical SMILES string or None if invalid
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available - cannot canonicalize SMILES")
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        return None
    except Exception:
        return None


def calculate_similarity(smiles1: str, smiles2: str, method: str = "tanimoto") -> float:
    """
    Calculate molecular similarity between two SMILES.
    
    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string
        method: Similarity method ('tanimoto', 'dice', 'cosine')
        
    Returns:
        Similarity score (0-1)
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available - cannot calculate similarity")
        return 0.0
    
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        # Calculate Morgan fingerprints
        fp1 = GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2 = GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        
        # Calculate similarity
        if method.lower() == "tanimoto":
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif method.lower() == "dice":
            return DataStructs.DiceSimilarity(fp1, fp2)
        elif method.lower() == "cosine":
            return DataStructs.CosineSimilarity(fp1, fp2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
            
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0


def calculate_molecular_descriptors(smiles: str) -> Dict[str, float]:
    """
    Calculate molecular descriptors for a SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of molecular descriptors
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available - cannot calculate descriptors")
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        descriptors = {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_hbd': Descriptors.NumHDonors(mol),
            'num_hba': Descriptors.NumHAcceptors(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_rings': Descriptors.RingCount(mol),

            'molar_refractivity': Descriptors.MolMR(mol)
        }
        
        return descriptors
        
    except Exception as e:
        logger.error(f"Error calculating descriptors for {smiles}: {e}")
        return {}


def check_lipinski_rule_of_five(smiles: str) -> Dict[str, Any]:
    """
    Check Lipinski's Rule of Five for drug-likeness.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary with rule checks and violations
    """
    descriptors = calculate_molecular_descriptors(smiles)
    
    if not descriptors:
        return {"valid": False, "error": "Could not calculate descriptors"}
    
    # Lipinski's Rule of Five criteria
    mw_ok = descriptors.get('molecular_weight', 0) <= 500
    logp_ok = descriptors.get('logp', 0) <= 5
    hbd_ok = descriptors.get('num_hbd', 0) <= 5
    hba_ok = descriptors.get('num_hba', 0) <= 10
    
    violations = sum([not mw_ok, not logp_ok, not hbd_ok, not hba_ok])
    
    return {
        "valid": True,
        "molecular_weight": descriptors.get('molecular_weight', 0),
        "logp": descriptors.get('logp', 0),
        "num_hbd": descriptors.get('num_hbd', 0),
        "num_hba": descriptors.get('num_hba', 0),
        "mw_ok": mw_ok,
        "logp_ok": logp_ok,
        "hbd_ok": hbd_ok,
        "hba_ok": hba_ok,
        "violations": violations,
        "drug_like": violations <= 1  # Allow 1 violation
    }


def plot_molecular_properties(df: pd.DataFrame, properties: List[str] = None, 
                             save_path: Optional[str] = None):
    """
    Plot molecular property distributions.
    
    Args:
        df: DataFrame with molecular data
        properties: List of properties to plot
        save_path: Path to save the plot
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Matplotlib/Seaborn not available - cannot create plots")
        return
    
    if properties is None:
        properties = ['molecular_weight', 'logp', 'tpsa', 'num_hbd', 'num_hba']
    
    # Filter properties that exist in the dataframe
    available_properties = [prop for prop in properties if prop in df.columns]
    
    if not available_properties:
        logger.warning("No specified properties found in dataframe")
        return
    
    # Create subplots
    n_props = len(available_properties)
    n_cols = min(3, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_props == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, prop in enumerate(available_properties):
        if i < len(axes):
            sns.histplot(data=df, x=prop, ax=axes[i], kde=True)
            axes[i].set_title(f'Distribution of {prop}')
            axes[i].set_xlabel(prop.replace('_', ' ').title())
    
    # Hide unused subplots
    for i in range(len(available_properties), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, properties: List[str] = None,
                           save_path: Optional[str] = None):
    """
    Plot correlation matrix of molecular properties.
    
    Args:
        df: DataFrame with molecular data
        properties: List of properties to include
        save_path: Path to save the plot
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Matplotlib/Seaborn not available - cannot create plots")
        return
    
    if properties is None:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        properties = [col for col in numeric_cols if col not in ['smiles', 'canonical_smiles']]
    
    # Filter properties that exist in the dataframe
    available_properties = [prop for prop in properties if prop in df.columns]
    
    if len(available_properties) < 2:
        logger.warning("Need at least 2 properties for correlation matrix")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[available_properties].corr()
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Molecular Properties Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {save_path}")
    
    plt.show()


def explain_prediction(model, smiles: str, feature_names: List[str] = None) -> Any:
    """
    Explain model prediction using SHAP.
    
    Args:
        model: Trained model
        smiles: SMILES string to explain
        feature_names: List of feature names
        
    Returns:
        SHAP explanation object
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available - cannot generate explanations")
        return None
    
    try:
        # This is a placeholder implementation
        # In practice, you would:
        # 1. Convert SMILES to features
        # 2. Create SHAP explainer
        # 3. Generate explanations
        
        logger.info(f"Generating SHAP explanation for {smiles}")
        # Placeholder return
        return None
        
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return None


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif output_path.suffix.lower() == '.yaml':
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}")
    
    logger.info(f"Results saved to {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_output_directory(base_path: str, experiment_name: str = None) -> Path:
    """
    Create output directory for experiment results.
    
    Args:
        base_path: Base output path
        experiment_name: Name of the experiment
        
    Returns:
        Path to created directory
    """
    if experiment_name is None:
        from datetime import datetime
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(base_path) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def filter_molecules_by_properties(df: pd.DataFrame, filters: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Filter molecules based on property criteria.
    
    Args:
        df: DataFrame with molecular data
        filters: Dictionary of property filters
                Example: {'molecular_weight': {'min': 150, 'max': 500}}
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for property_name, criteria in filters.items():
        if property_name not in df.columns:
            logger.warning(f"Property {property_name} not found in dataframe")
            continue
        
        if 'min' in criteria:
            filtered_df = filtered_df[filtered_df[property_name] >= criteria['min']]
        
        if 'max' in criteria:
            filtered_df = filtered_df[filtered_df[property_name] <= criteria['max']]
    
    logger.info(f"Filtered {len(df)} molecules to {len(filtered_df)} molecules")
    return filtered_df


def calculate_diversity_metrics(smiles_list: List[str]) -> Dict[str, float]:
    """
    Calculate diversity metrics for a set of molecules.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Dictionary of diversity metrics
    """
    if not RDKIT_AVAILABLE:
        logger.warning("RDKit not available - cannot calculate diversity")
        return {}
    
    try:
        # Calculate pairwise similarities
        similarities = []
        valid_molecules = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_molecules.append(smiles)
        
        if len(valid_molecules) < 2:
            return {"error": "Need at least 2 valid molecules"}
        
        for i in range(len(valid_molecules)):
            for j in range(i + 1, len(valid_molecules)):
                sim = calculate_similarity(valid_molecules[i], valid_molecules[j])
                similarities.append(sim)
        
        if not similarities:
            return {"error": "Could not calculate similarities"}
        
        similarities = np.array(similarities)
        
        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "diversity_score": float(1 - np.mean(similarities)),  # 1 - mean similarity
            "num_molecules": len(valid_molecules),
            "num_comparisons": len(similarities)
        }
        
    except Exception as e:
        logger.error(f"Error calculating diversity metrics: {e}")
        return {"error": str(e)}


def benchmark_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                               task_type: str = "regression") -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        task_type: 'regression' or 'classification'
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score
        )
        
        metrics = {}
        
        if task_type == "regression":
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            # Additional regression metrics
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['max_error'] = np.max(np.abs(y_true - y_pred))
            
        elif task_type == "classification":
            # Convert to binary if needed
            if len(np.unique(y_pred)) > 2:
                y_pred_binary = (y_pred > 0.5).astype(int)
            else:
                y_pred_binary = y_pred
            
            metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
            metrics['precision'] = precision_score(y_true, y_pred_binary, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred_binary, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred_binary, average='weighted', zero_division=0)
            
            # AUC if binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    metrics['auc'] = roc_auc_score(y_true, y_pred)
                except ValueError:
                    pass
        
        return metrics
        
    except ImportError:
        logger.warning("Scikit-learn not available - limited metrics")
        return {}
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}


def assess_data_quality(df: pd.DataFrame, smiles_column: str = 'smiles') -> Dict[str, Any]:
    """
    Comprehensive data quality assessment for molecular datasets.

    Args:
        df: DataFrame to assess
        smiles_column: Name of SMILES column

    Returns:
        Dictionary with quality assessment results
    """
    quality_report = {
        'dataset_size': len(df),
        'columns': list(df.columns),
        'missing_data': {},
        'data_types': {},
        'smiles_quality': {},
        'duplicates': {},
        'outliers': {},
        'recommendations': []
    }

    # Missing data analysis
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        quality_report['missing_data'][col] = {
            'count': missing_count,
            'percentage': missing_pct
        }

        if missing_pct > 50:
            quality_report['recommendations'].append(f"Consider removing column '{col}' (>{missing_pct:.1f}% missing)")
        elif missing_pct > 20:
            quality_report['recommendations'].append(f"High missing data in '{col}' ({missing_pct:.1f}%)")

    # Data types analysis
    for col in df.columns:
        quality_report['data_types'][col] = str(df[col].dtype)

    # SMILES quality assessment
    if smiles_column in df.columns:
        smiles_data = df[smiles_column].dropna()

        if RDKIT_AVAILABLE:
            valid_smiles = 0
            invalid_smiles = []

            for idx, smiles in enumerate(smiles_data):
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    valid_smiles += 1
                else:
                    invalid_smiles.append((idx, smiles))

            quality_report['smiles_quality'] = {
                'total_smiles': len(smiles_data),
                'valid_smiles': valid_smiles,
                'invalid_smiles': len(invalid_smiles),
                'validity_rate': (valid_smiles / len(smiles_data)) * 100 if smiles_data.any() else 0,
                'invalid_examples': invalid_smiles[:5]  # First 5 invalid examples
            }

            if len(invalid_smiles) > 0:
                quality_report['recommendations'].append(f"Remove {len(invalid_smiles)} invalid SMILES")
        else:
            quality_report['smiles_quality'] = {'note': 'RDKit not available for SMILES validation'}

    # Duplicate analysis
    total_duplicates = df.duplicated().sum()
    quality_report['duplicates']['total_duplicate_rows'] = total_duplicates

    if smiles_column in df.columns:
        smiles_duplicates = df.duplicated(subset=[smiles_column]).sum()
        quality_report['duplicates']['duplicate_smiles'] = smiles_duplicates

        if smiles_duplicates > 0:
            quality_report['recommendations'].append(f"Remove {smiles_duplicates} duplicate SMILES")

    # Outlier detection for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = data[(data < lower_bound) | (data > upper_bound)]
                quality_report['outliers'][col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(data)) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }

    # Overall quality score
    quality_score = 100
    if quality_report['missing_data']:
        avg_missing = np.mean([v['percentage'] for v in quality_report['missing_data'].values()])
        quality_score -= avg_missing * 0.5

    if 'validity_rate' in quality_report['smiles_quality']:
        validity_rate = quality_report['smiles_quality']['validity_rate']
        quality_score -= (100 - validity_rate) * 0.3

    quality_report['overall_quality_score'] = max(0, quality_score)

    return quality_report


def create_progress_tracker(total_items: int, description: str = "Processing") -> Any:
    """
    Create a progress tracker for long-running operations.

    Args:
        total_items: Total number of items to process
        description: Description of the operation

    Returns:
        Progress tracker object
    """
    try:
        return tqdm(total=total_items, desc=description)
    except NameError:
        # Fallback if tqdm is not available
        class SimpleProgressTracker:
            def __init__(self, total, desc):
                self.total = total
                self.desc = desc
                self.current = 0

            def update(self, n=1):
                self.current += n
                if self.current % max(1, self.total // 10) == 0:
                    print(f"{self.desc}: {self.current}/{self.total} ({self.current/self.total*100:.1f}%)")

            def close(self):
                print(f"{self.desc}: Complete ({self.total}/{self.total})")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        return SimpleProgressTracker(total_items, description)


def visualize_molecular_dataset(df: pd.DataFrame, smiles_column: str = 'smiles',
                               target_column: str = None, save_path: str = None):
    """
    Create comprehensive visualizations for molecular datasets.

    Args:
        df: DataFrame with molecular data
        smiles_column: Name of SMILES column
        target_column: Name of target column for analysis
        save_path: Path to save visualizations
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Matplotlib/Seaborn not available - cannot create visualizations")
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Molecular Dataset Analysis', fontsize=16)

    # 1. Missing data heatmap
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_data.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Missing Data by Column')
        axes[0, 0].set_ylabel('Missing Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
    else:
        axes[0, 0].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Missing Data by Column')

    # 2. Data types distribution
    dtype_counts = df.dtypes.value_counts()
    axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Data Types Distribution')

    # 3. Dataset size info
    axes[0, 2].text(0.1, 0.8, f'Total Records: {len(df):,}', transform=axes[0, 2].transAxes, fontsize=12)
    axes[0, 2].text(0.1, 0.6, f'Total Columns: {len(df.columns)}', transform=axes[0, 2].transAxes, fontsize=12)
    axes[0, 2].text(0.1, 0.4, f'Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB',
                    transform=axes[0, 2].transAxes, fontsize=12)
    axes[0, 2].set_title('Dataset Information')
    axes[0, 2].axis('off')

    # 4. Molecular weight distribution (if available)
    if 'molecular_weight' in df.columns:
        df['molecular_weight'].hist(bins=50, ax=axes[1, 0])
        axes[1, 0].set_title('Molecular Weight Distribution')
        axes[1, 0].set_xlabel('Molecular Weight (Da)')
        axes[1, 0].set_ylabel('Frequency')
    else:
        axes[1, 0].text(0.5, 0.5, 'Molecular Weight\nNot Available', ha='center', va='center',
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Molecular Weight Distribution')

    # 5. Target distribution (if available)
    if target_column and target_column in df.columns:
        target_data = df[target_column].dropna()
        if target_data.dtype in ['int64', 'float64']:
            target_data.hist(bins=30, ax=axes[1, 1])
            axes[1, 1].set_title(f'{target_column} Distribution')
            axes[1, 1].set_xlabel(target_column)
            axes[1, 1].set_ylabel('Frequency')
        else:
            target_data.value_counts().plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title(f'{target_column} Distribution')
            axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'Target Column\nNot Specified', ha='center', va='center',
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Target Distribution')

    # 6. Correlation heatmap for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Correlation Matrix')
    else:
        axes[1, 2].text(0.5, 0.5, 'Insufficient Numeric\nColumns for Correlation',
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Correlation Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dataset visualization saved to {save_path}")

    plt.show()


def generate_data_report(df: pd.DataFrame, smiles_column: str = 'smiles',
                        target_column: str = None, output_path: str = None) -> str:
    """
    Generate a comprehensive data quality report.

    Args:
        df: DataFrame to analyze
        smiles_column: Name of SMILES column
        target_column: Name of target column
        output_path: Path to save the report

    Returns:
        Report as string
    """
    quality_assessment = assess_data_quality(df, smiles_column)

    report = []
    report.append("=" * 60)
    report.append("MOLECULAR DATASET QUALITY REPORT")
    report.append("=" * 60)
    report.append("")

    # Dataset overview
    report.append("DATASET OVERVIEW")
    report.append("-" * 20)
    report.append(f"Total records: {quality_assessment['dataset_size']:,}")
    report.append(f"Total columns: {len(quality_assessment['columns'])}")
    report.append(f"Overall quality score: {quality_assessment['overall_quality_score']:.1f}/100")
    report.append("")

    # Missing data summary
    report.append("MISSING DATA SUMMARY")
    report.append("-" * 20)
    missing_data = quality_assessment['missing_data']
    for col, info in missing_data.items():
        if info['count'] > 0:
            report.append(f"{col}: {info['count']} missing ({info['percentage']:.1f}%)")
    report.append("")

    # SMILES quality
    if 'smiles_quality' in quality_assessment and 'validity_rate' in quality_assessment['smiles_quality']:
        smiles_quality = quality_assessment['smiles_quality']
        report.append("SMILES QUALITY")
        report.append("-" * 15)
        report.append(f"Total SMILES: {smiles_quality['total_smiles']:,}")
        report.append(f"Valid SMILES: {smiles_quality['valid_smiles']:,}")
        report.append(f"Invalid SMILES: {smiles_quality['invalid_smiles']:,}")
        report.append(f"Validity rate: {smiles_quality['validity_rate']:.1f}%")
        report.append("")

    # Duplicates
    duplicates = quality_assessment['duplicates']
    report.append("DUPLICATE ANALYSIS")
    report.append("-" * 18)
    report.append(f"Duplicate rows: {duplicates['total_duplicate_rows']:,}")
    if 'duplicate_smiles' in duplicates:
        report.append(f"Duplicate SMILES: {duplicates['duplicate_smiles']:,}")
    report.append("")

    # Recommendations
    if quality_assessment['recommendations']:
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)
        for i, rec in enumerate(quality_assessment['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append("")

    # Data types
    report.append("COLUMN DATA TYPES")
    report.append("-" * 17)
    for col, dtype in quality_assessment['data_types'].items():
        report.append(f"{col}: {dtype}")

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        logger.info(f"Data quality report saved to {output_path}")

    return report_text


def main():
    """Example usage of utility functions."""
    # Test SMILES validation
    test_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "invalid_smiles"]
    
    print("Testing SMILES validation:")
    for smiles in test_smiles:
        is_valid = validate_smiles(smiles)
        canonical = canonicalize_smiles(smiles)
        print(f"  {smiles}: Valid={is_valid}, Canonical={canonical}")
    
    # Test molecular descriptors
    if RDKIT_AVAILABLE:
        print("\nTesting molecular descriptors:")
        descriptors = calculate_molecular_descriptors("CCO")
        for key, value in descriptors.items():
            print(f"  {key}: {value:.3f}")
        
        # Test Lipinski's rule
        print("\nTesting Lipinski's Rule of Five:")
        lipinski = check_lipinski_rule_of_five("CCO")
        print(f"  Drug-like: {lipinski.get('drug_like', False)}")
        print(f"  Violations: {lipinski.get('violations', 0)}")
    
    # Test similarity calculation
    if RDKIT_AVAILABLE:
        print("\nTesting similarity calculation:")
        sim = calculate_similarity("CCO", "CC(=O)O")
        print(f"  Similarity between CCO and CC(=O)O: {sim:.3f}")
    
    # Test diversity metrics
    if RDKIT_AVAILABLE:
        print("\nTesting diversity metrics:")
        diversity = calculate_diversity_metrics(["CCO", "CC(=O)O", "c1ccccc1"])
        print(f"  Diversity score: {diversity.get('diversity_score', 'N/A')}")
    
    print("\nUtility functions testing completed!")


if __name__ == "__main__":
    main()