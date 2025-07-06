"""
API Models for Drug Discovery Compound Optimization

This module contains Pydantic models for request/response validation,
molecular property prediction schemas, and error handling.
"""

from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create dummy base classes
    class BaseModel:
        pass
    class Field:
        pass
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def model_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SimilarityMetric(str, Enum):
    """Molecular similarity metrics."""
    TANIMOTO = "tanimoto"
    DICE = "dice"
    COSINE = "cosine"
    JACCARD = "jaccard"


class PropertyType(str, Enum):
    """Types of molecular properties."""
    MOLECULAR_WEIGHT = "molecular_weight"
    LOGP = "logp"
    TPSA = "tpsa"
    NUM_ROTATABLE_BONDS = "num_rotatable_bonds"
    NUM_HBD = "num_hbd"
    NUM_HBA = "num_hba"
    NUM_AROMATIC_RINGS = "num_aromatic_rings"
    SOLUBILITY = "solubility"
    BIOACTIVITY = "bioactivity"
    TOXICITY = "toxicity"
    DRUG_LIKENESS = "drug_likeness"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    message: str = "Success"
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None

    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        super().__init__(**data)

class ErrorResponse(BaseResponse):
    """Error response model."""
    success: bool = False
    error_code: str
    error_details: Optional[Dict[str, Any]] = None


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseResponse):
    """Paginated response model."""
    total_items: int
    total_pages: int
    current_page: int
    page_size: int
    has_next: bool
    has_previous: bool


# SMILES and Molecular Input Models
class SMILESInput(BaseModel):
    """Input model for single SMILES string."""
    smiles: str = Field(
        min_length=1, max_length=1000,
        description="SMILES string representing the molecule"
    )
    
    @field_validator('smiles')
    @classmethod
    def validate_smiles_format(cls, v):
        """Basic SMILES format validation."""
        if not v or not v.strip():
            raise ValueError("SMILES string cannot be empty")
        
        # Basic character validation
        allowed_chars = set('CNOPSFClBrI()[]{}=-#@+.:0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if not all(c in allowed_chars for c in v):
            raise ValueError("SMILES contains invalid characters")
        
        return v.strip()


class BatchSMILESInput(BaseModel):
    """Input model for batch SMILES processing."""
    smiles_list: List[str] = Field(
        min_length=1, max_length=1000,
        description="List of SMILES strings"
    )
    
    @field_validator('smiles_list')
    @classmethod
    def validate_smiles_list(cls, v):
        """Validate list of SMILES strings."""
        if not v:
            raise ValueError("SMILES list cannot be empty")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_smiles = []
        for smiles in v:
            if smiles not in seen:
                seen.add(smiles)
                unique_smiles.append(smiles.strip())
        
        return unique_smiles


class MolecularFileUpload(BaseModel):
    """Model for molecular file upload metadata."""
    filename: str
    file_format: str = Field(description="csv, sdf, xlsx")
    smiles_column: str = Field(default="smiles", description="Name of SMILES column")
    target_column: Optional[str] = Field(default=None, description="Target property column")
    has_header: bool = Field(default=True, description="File has header row")
    
    @field_validator('file_format')
    @classmethod
    def validate_file_format(cls, v):
        """Validate file format."""
        allowed_formats = ['csv', 'sdf', 'xlsx', 'xls']
        if v.lower() not in allowed_formats:
            raise ValueError(f"File format must be one of: {allowed_formats}")
        return v.lower()


# Property Prediction Models
class PropertyPredictionRequest(BaseModel):
    """Request model for property prediction."""
    smiles: Union[str, List[str]] = Field(description="SMILES string or list of SMILES")
    properties: List[PropertyType] = Field(
        default=[PropertyType.MOLECULAR_WEIGHT, PropertyType.LOGP, PropertyType.TPSA],
        description="List of properties to predict"
    )
    include_descriptors: bool = Field(default=True, description="Include molecular descriptors")
    include_fingerprints: bool = Field(default=False, description="Include molecular fingerprints")
    
    @field_validator('smiles')
    @classmethod
    def validate_smiles_input(cls, v):
        """Validate SMILES input."""
        if isinstance(v, str):
            return [v.strip()]
        elif isinstance(v, list):
            return [s.strip() for s in v if s and isinstance(s, str)]
        else:
            raise ValueError("SMILES must be a string or list of strings")


class MolecularProperty(BaseModel):
    """Individual molecular property."""
    name: str
    value: float
    unit: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    method: Optional[str] = None


class MolecularDescriptors(BaseModel):
    """Molecular descriptors."""
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    num_rotatable_bonds: Optional[int] = None
    num_hbd: Optional[int] = None
    num_hba: Optional[int] = None
    num_aromatic_rings: Optional[int] = None
    num_heavy_atoms: Optional[int] = None
    num_rings: Optional[int] = None
    molar_refractivity: Optional[float] = None


class DrugLikenessAssessment(BaseModel):
    """Drug-likeness assessment based on various rules."""
    lipinski_violations: int = Field(ge=0, le=4)
    lipinski_compliant: bool
    veber_compliant: bool
    drug_like_score: float = Field(ge=0.0, le=1.0)
    alerts: List[str] = Field(default_factory=list)


class PropertyPredictionResult(BaseModel):
    """Result for single molecule property prediction."""
    smiles: str
    canonical_smiles: Optional[str] = None
    valid: bool
    properties: Dict[str, float] = Field(default_factory=dict)
    descriptors: Optional[MolecularDescriptors] = None
    fingerprints: Optional[Dict[str, Any]] = None
    drug_likeness: Optional[DrugLikenessAssessment] = None
    processing_time: Optional[float] = None
    errors: List[str] = Field(default_factory=list)


class PropertyPredictionResponse(BaseResponse):
    """Response model for property prediction."""
    results: List[PropertyPredictionResult]
    total_molecules: int
    valid_molecules: int
    invalid_molecules: int
    processing_time: float


# Similarity Calculation Models
class SimilarityRequest(BaseModel):
    """Request model for molecular similarity calculation."""
    query_smiles: str = Field(min_length=1, description="Query molecule SMILES")
    target_smiles: List[str] = Field(
        min_length=1, max_length=1000,
        description="Target molecules SMILES list"
    )
    similarity_metric: SimilarityMetric = Field(
        default=SimilarityMetric.TANIMOTO,
        description="Similarity metric to use"
    )
    threshold: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum similarity threshold"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=100,
        description="Return top K most similar molecules"
    )


class SimilarityResult(BaseModel):
    """Single similarity calculation result."""
    target_smiles: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    metric: SimilarityMetric
    rank: Optional[int] = None


class SimilarityResponse(BaseResponse):
    """Response model for similarity calculation."""
    query_smiles: str
    results: List[SimilarityResult]
    total_comparisons: int
    above_threshold: int
    metric: SimilarityMetric
    processing_time: float


# Compound Optimization Models
class OptimizationTarget(BaseModel):
    """Optimization target specification."""
    property_name: PropertyType
    target_value: float
    weight: float = Field(default=1.0, ge=0.0, description="Importance weight")
    tolerance: float = Field(default=0.1, ge=0.0, description="Acceptable deviation")
    direction: str = Field(default="target", description="minimize, maximize, or target")
    
    @field_validator('direction')
    @classmethod
    def validate_direction(cls, v):
        """Validate optimization direction."""
        allowed_directions = ['minimize', 'maximize', 'target']
        if v not in allowed_directions:
            raise ValueError(f"Direction must be one of: {allowed_directions}")
        return v


class OptimizationConstraints(BaseModel):
    """Optimization constraints."""
    max_heavy_atoms: Optional[int] = Field(default=None, ge=1, le=100)
    max_molecular_weight: Optional[float] = Field(default=None, ge=0.0)
    min_molecular_weight: Optional[float] = Field(default=None, ge=0.0)
    preserve_scaffold: bool = Field(default=False, description="Preserve molecular scaffold")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    drug_like_only: bool = Field(default=True, description="Only generate drug-like molecules")


class OptimizationRequest(BaseModel):
    """Request model for compound optimization."""
    starting_smiles: str = Field(min_length=1, description="Starting molecule SMILES")
    targets: List[OptimizationTarget] = Field(
        min_length=1, max_length=10,
        description="Optimization targets"
    )
    constraints: OptimizationConstraints = Field(
        default_factory=OptimizationConstraints,
        description="Optimization constraints"
    )
    max_iterations: int = Field(
        default=100, ge=1, le=1000,
        description="Maximum optimization iterations"
    )
    population_size: int = Field(
        default=50, ge=1, le=200,
        description="Population size for genetic algorithm"
    )
    convergence_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Convergence threshold"
    )
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class OptimizationCandidate(BaseModel):
    """Optimization candidate result."""
    smiles: str
    canonical_smiles: str
    generation: int
    fitness_score: float
    properties: Dict[str, float]
    similarity_to_parent: float
    violations: List[str] = Field(default_factory=list)


class OptimizationResult(BaseModel):
    """Optimization process result."""
    task_id: str
    status: TaskStatus
    starting_smiles: str
    best_candidates: List[OptimizationCandidate]
    total_generations: int
    convergence_achieved: bool
    final_fitness_score: float
    processing_time: float
    error_message: Optional[str] = None


class OptimizationResponse(BaseResponse):
    """Response model for optimization request."""
    task_id: str
    status: TaskStatus
    estimated_completion_time: Optional[int] = None  # seconds
    polling_url: Optional[str] = None


# Batch Processing Models
class BatchJobRequest(BaseModel):
    """Request model for batch processing jobs."""
    job_name: str = Field(description="Human-readable job name")
    job_type: str = Field(description="Type of batch job")
    input_data: Dict[str, Any] = Field(description="Input data for processing")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Job parameters")
    priority: int = Field(default=5, ge=1, le=10, description="Job priority (1=highest)")
    
    @field_validator('job_type')
    @classmethod
    def validate_job_type(cls, v):
        """Validate batch job type."""
        allowed_types = ['property_prediction', 'similarity_search', 'optimization', 'feature_extraction']
        if v not in allowed_types:
            raise ValueError(f"Job type must be one of: {allowed_types}")
        return v


class BatchJobStatus(BaseModel):
    """Batch job status information."""
    job_id: str
    job_name: str
    job_type: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(ge=0.0, le=1.0, description="Job completion progress")
    total_items: int
    processed_items: int
    failed_items: int
    current_stage: str
    error_message: Optional[str] = None
    result_url: Optional[str] = None


class BatchJobResponse(BaseResponse):
    """Response model for batch job submission."""
    job_id: str
    status: TaskStatus
    estimated_completion_time: Optional[int] = None
    polling_url: str
    result_url: Optional[str] = None


# Health Check and System Models
class SystemHealth(BaseModel):
    """System health status."""
    status: str = Field(description="overall, healthy, degraded, unhealthy")
    version: str
    uptime: int = Field(description="Uptime in seconds")
    timestamp: datetime


class ComponentHealth(BaseModel):
    """Individual component health."""
    name: str
    status: str = Field(description="healthy, degraded, unhealthy")
    response_time: Optional[float] = None
    error_message: Optional[str] = None
    last_check: datetime


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    version: str
    type: str
    loaded: bool
    load_time: Optional[datetime] = None
    memory_usage: Optional[float] = None  # MB
    accuracy: Optional[float] = None
    training_date: Optional[datetime] = None


class HealthResponse(BaseResponse):
    """Complete health check response."""
    system: SystemHealth
    components: List[ComponentHealth]
    models: List[ModelInfo]
    dependencies: Dict[str, bool]
    performance_metrics: Dict[str, float]


# API Usage and Analytics Models
class APIUsageStats(BaseModel):
    """API usage statistics."""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    peak_response_time: float
    requests_per_minute: float
    last_request: datetime


class UserQuota(BaseModel):
    """User quota information."""
    user_id: str
    requests_used: int
    requests_limit: int
    requests_remaining: int
    reset_time: datetime
    rate_limit_per_minute: int


class APIMetricsResponse(BaseResponse):
    """API metrics response."""
    usage_stats: List[APIUsageStats]
    total_requests: int
    uptime: int
    error_rate: float
    average_response_time: float


# Data Upload and Processing Models
class DataUploadRequest(BaseModel):
    """Data upload request metadata."""
    filename: str
    file_size: int = Field(description="File size in bytes")
    file_format: str
    smiles_column: str = Field(default="smiles")
    target_columns: List[str] = Field(default_factory=list)
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v):
        """Validate file size (max 100MB)."""
        max_size = 100 * 1024 * 1024  # 100MB
        if v > max_size:
            raise ValueError(f"File size cannot exceed {max_size} bytes")
        return v


class DataProcessingResult(BaseModel):
    """Data processing result."""
    total_rows: int
    valid_rows: int
    invalid_rows: int
    processing_time: float
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    output_file_url: Optional[str] = None


class DataUploadResponse(BaseResponse):
    """Data upload response."""
    upload_id: str
    filename: str
    processing_result: DataProcessingResult
    download_url: Optional[str] = None


# Configuration Models
class APIConfig(BaseModel):
    """API configuration model."""
    rate_limit_per_minute: int = Field(default=100, ge=1, le=10000)
    max_batch_size: int = Field(default=1000, ge=1, le=10000)
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    cache_ttl_seconds: int = Field(default=3600, ge=0, le=86400)
    enable_swagger: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING, ERROR")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()


# Export all models
__all__ = [
    # Enums
    'TaskStatus', 'SimilarityMetric', 'PropertyType',
    
    # Base models
    'BaseResponse', 'ErrorResponse', 'PaginationParams', 'PaginatedResponse',
    
    # Input models
    'SMILESInput', 'BatchSMILESInput', 'MolecularFileUpload',
    
    # Property prediction models
    'PropertyPredictionRequest', 'MolecularProperty', 'MolecularDescriptors',
    'DrugLikenessAssessment', 'PropertyPredictionResult', 'PropertyPredictionResponse',
    
    # Similarity models
    'SimilarityRequest', 'SimilarityResult', 'SimilarityResponse',
    
    # Optimization models
    'OptimizationTarget', 'OptimizationConstraints', 'OptimizationRequest',
    'OptimizationCandidate', 'OptimizationResult', 'OptimizationResponse',
    
    # Batch processing models
    'BatchJobRequest', 'BatchJobStatus', 'BatchJobResponse',
    
    # Health and system models
    'SystemHealth', 'ComponentHealth', 'ModelInfo', 'HealthResponse',
    
    # Analytics models
    'APIUsageStats', 'UserQuota', 'APIMetricsResponse',
    
    # Data upload models
    'DataUploadRequest', 'DataProcessingResult', 'DataUploadResponse',
    
    # Configuration models
    'APIConfig'
]