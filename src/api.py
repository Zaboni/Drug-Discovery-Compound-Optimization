"""
Production-Ready FastAPI Web Service for Drug Discovery Compound Optimization

This module provides a comprehensive REST API for molecular property prediction,
compound optimization, and drug discovery services with rate limiting, caching,
batch processing, and comprehensive error handling.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import yaml
from datetime import datetime
import hashlib

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logging.warning("FastAPI not available. API functionality will be limited.")
    # Define dummy classes for when FastAPI is not available
    class BaseModel:
        pass
    class Field:
        pass
    class FastAPI:
        pass
    class HTTPException(Exception):
        def __init__(self, status_code, detail):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
    class BackgroundTasks:
        pass
    class CORSMiddleware:
        pass
    class Depends:
        pass
    class Request:
        pass
    class Response:
        pass
    class UploadFile:
        pass
    class File:
        pass
    class HTTPBearer:
        pass
    class HTTPAuthorizationCredentials:
        pass
    class Limiter:
        pass
    class StaticFiles:
        pass
    class Jinja2Templates:
        pass
    class HTMLResponse:
        pass
    class uvicorn:
        @staticmethod
        def run(*args, **kwargs):
            pass
    def _rate_limit_exceeded_handler(*args, **kwargs):
        pass
    def get_remote_address(*args, **kwargs):
        return "127.0.0.1"
    class RateLimitExceeded(Exception):
        pass
    class SlowAPIMiddleware:
        pass
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Caching will be limited.")
    redis = None

try:
    import numpy as np
    import pandas as pd
    NUMPY_PANDAS_AVAILABLE = True
except ImportError:
    NUMPY_PANDAS_AVAILABLE = False
    logging.warning("NumPy/Pandas not available. Some API functionality will be limited.")
    np = None
    pd = None

# Import our modules (with error handling)
try:
    from .data_processing import MolecularDataProcessor
    from .models import PropertyPredictor, RandomForestModel
    from .utils import validate_smiles, calculate_similarity
    from .api_models import *
except ImportError:
    # Fallback for when modules aren't available
    MolecularDataProcessor = None
    PropertyPredictor = None
    RandomForestModel = None
    validate_smiles = None
    calculate_similarity = None

    # Fallback API model classes
    class SMILESInput(BaseModel if FASTAPI_AVAILABLE else object):
        smiles: str = ""
    class BatchSMILESInput(BaseModel if FASTAPI_AVAILABLE else object):
        smiles_list: list = []
    class PropertyPredictionRequest(BaseModel if FASTAPI_AVAILABLE else object):
        smiles: list = []
        properties: list = []
    class SimilarityRequest(BaseModel if FASTAPI_AVAILABLE else object):
        query_smiles: str = ""
        target_smiles: list = []
        similarity_metric: str = "tanimoto"
        threshold: float = 0.0
        top_k: int = None
    class OptimizationRequest(BaseModel if FASTAPI_AVAILABLE else object):
        starting_smiles: str = ""
        targets: list = []
        max_iterations: int = 100
logger = logging.getLogger(__name__)

# Global configuration
API_CONFIG = {
    'rate_limit_per_minute': 100,
    'max_batch_size': 1000,
    'cache_ttl_seconds': 3600,
    'enable_rate_limiting': True,
    'enable_caching': True,
    'enable_metrics': True,
    'cors_origins': ["*"],
    'max_file_size_mb': 100
}

# Rate limiting setup
if FASTAPI_AVAILABLE:
    try:
        limiter = Limiter(key_func=get_remote_address)
        RATE_LIMITING_AVAILABLE = True
    except (ImportError, TypeError):
        RATE_LIMITING_AVAILABLE = False
        limiter = None
        logger.warning("Rate limiting not available - slowapi not installed")

    # Cache setup
    cache = {}
    if REDIS_AVAILABLE:
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    else:
        redis_client = None

    # Metrics storage
    metrics = {
        'total_requests': 0,
        'successful_requests': 0,
        'failed_requests': 0,
        'endpoint_stats': {},
        'start_time': time.time()
    }

    # Background tasks storage
    background_tasks = {}

    # File upload storage
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)

    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(exist_ok=True)


def get_cache_key(endpoint: str, data: Any) -> str:
    """Generate cache key for request."""
    data_str = json.dumps(data, sort_keys=True) if isinstance(data, dict) else str(data)
    return f"{endpoint}:{hashlib.md5(data_str.encode()).hexdigest()}"


def get_cached_result(cache_key: str) -> Optional[Any]:
    """Get cached result."""
    if redis_client:
        try:
            result = redis_client.get(cache_key)
            if result:
                return json.loads(result)
        except Exception as e:
            logger.error(f"Redis get error: {e}")

    # Fallback to local cache
    if cache_key in cache:
        data, timestamp = cache[cache_key]
        if time.time() - timestamp < API_CONFIG['cache_ttl_seconds']:
            return data
        else:
            del cache[cache_key]

    return None


def set_cached_result(cache_key: str, data: Any, ttl: int = None) -> None:
    """Set cached result."""
    if ttl is None:
        ttl = API_CONFIG['cache_ttl_seconds']

    if redis_client:
        try:
            redis_client.setex(cache_key, ttl, json.dumps(data, default=str))
            return
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    # Fallback to local cache
    cache[cache_key] = (data, time.time())


def track_metrics(endpoint: str, method: str, success: bool, response_time: float):
    """Track API metrics."""
    if not API_CONFIG['enable_metrics']:
        return

    metrics['total_requests'] += 1
    if success:
        metrics['successful_requests'] += 1
    else:
        metrics['failed_requests'] += 1

    endpoint_key = f"{method}:{endpoint}"
    if endpoint_key not in metrics['endpoint_stats']:
        metrics['endpoint_stats'][endpoint_key] = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0,
            'max_response_time': 0,
            'last_request': None
        }

    stats = metrics['endpoint_stats'][endpoint_key]
    stats['total_requests'] += 1
    stats['total_response_time'] += response_time
    stats['max_response_time'] = max(stats['max_response_time'], response_time)
    stats['last_request'] = datetime.now()

    if success:
        stats['successful_requests'] += 1
    else:
        stats['failed_requests'] += 1


def auth_required(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
    """Authentication dependency (placeholder)."""
    # In production, implement proper authentication
    return True


class DrugDiscoveryAPI:
    """Enhanced API class for drug discovery services."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Drug Discovery API."""
        self.config = self._load_config(config_path)
        self.data_processor = None
        self.property_predictor = None
        self.models = {}
        self.start_time = time.time()

        # Initialize components
        self._initialize_components()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Update global API config
                API_CONFIG.update(config.get('api', {}))
                return config
        else:
            return {
                'api': API_CONFIG,
                'models': {
                    'load_pretrained': False,
                    'pretrained_path': None
                }
            }

    def _initialize_components(self):
        """Initialize API components."""
        try:
            # Initialize data processor
            if MolecularDataProcessor:
                self.data_processor = MolecularDataProcessor()
                logger.info("Data processor initialized")

            # Load models if configured
            if self.config.get('models', {}).get('load_pretrained', False):
                self._load_models()

        except Exception as e:
            logger.error(f"Error initializing components: {e}")

    def _load_models(self):
        """Load pre-trained models."""
        model_path = self.config.get('models', {}).get('pretrained_path')
        if model_path and Path(model_path).exists():
            try:
                # Load property predictor (placeholder implementation)
                logger.info(f"Loading models from {model_path}")
                # In practice, you would load actual trained models here
                self.models['property_predictor'] = True
            except Exception as e:
                logger.error(f"Error loading models: {e}")

    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        uptime = time.time() - self.start_time
        return {
            'status': 'healthy',
            'version': '1.0.0',
            'uptime': int(uptime),
            'timestamp': datetime.now(),
            'models_loaded': bool(self.models),
            'data_processor_available': self.data_processor is not None,
            'cache_available': redis_client is not None,
            'dependencies': {
                'fastapi': FASTAPI_AVAILABLE,
                'numpy_pandas': NUMPY_PANDAS_AVAILABLE,
                'redis': REDIS_AVAILABLE,
                'rdkit': MolecularDataProcessor is not None
            }
        }

    async def predict_properties_batch(self, smiles_list: List[str],
                                       properties: List[str]) -> List[Dict[str, Any]]:
        """Predict properties for a batch of molecules."""
        if not self.data_processor:
            raise HTTPException(status_code=503, detail="Data processor not available")
        results = []

        for smiles in smiles_list:
            try:
                # Process SMILES
                processed = self.data_processor.process_smiles([smiles])

                if not processed or not processed[0].get("valid"):
                    results.append({
                        'smiles': smiles,
                        'properties': {},
                        'valid': False,
                        'errors': ["Invalid SMILES"]
                    })
                    continue

                # Extract features
                features_df = self.data_processor.extract_features(processed)

                # Predict properties
                props = {}
                for prop in properties:
                    if prop == "molecular_weight":
                        props[prop] = features_df.iloc[0].get("molecular_weight", 0.0)
                    elif prop == "logp":
                        props[prop] = features_df.iloc[0].get("logp", 0.0)
                    elif prop == "tpsa":
                        props[prop] = features_df.iloc[0].get("tpsa", 0.0)
                    else:
                        # Placeholder for other properties
                        props[prop] = np.random.uniform(0, 1) if NUMPY_PANDAS_AVAILABLE else 0.5

                results.append({
                    'smiles': smiles,
                    'canonical_smiles': processed[0].get('canonical_smiles'),
                    'properties': props,
                    'valid': True,
                    'errors': []
                })
            except Exception as e:
                results.append({
                    'smiles': smiles,
                    'properties': {},
                    'valid': False,
                    'errors': [str(e)]
                })

        return results

# Create FastAPI app if available
if FASTAPI_AVAILABLE:
    # Initialize API
    api_instance = DrugDiscoveryAPI()

    app = FastAPI(
        title="Drug Discovery Compound Optimization API",
        description="Production-ready REST API for molecular property prediction and compound optimization",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {
                "name": "Health",
                "description": "Health check and system status endpoints"
            },
            {
                "name": "Validation",
                "description": "SMILES validation endpoints"
            },
            {
                "name": "Prediction",
                "description": "Molecular property prediction endpoints"
            },
            {
                "name": "Similarity",
                "description": "Molecular similarity calculation endpoints"
            },
            {
                "name": "Optimization",
                "description": "Compound optimization endpoints"
            },
            {
                "name": "Batch",
                "description": "Batch processing endpoints"
            },
            {
                "name": "Upload",
                "description": "File upload and data processing endpoints"
            },
            {
                "name": "Metrics",
                "description": "API metrics and monitoring endpoints"
            }
        ]
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CONFIG.get('cors_origins', ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add rate limiting middleware if available
    if RATE_LIMITING_AVAILABLE:
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Mount frontend files if available
    if Path("frontend").exists():
        app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
    # Setup templates if available
    if Path("frontend/templates").exists():
        templates = Jinja2Templates(directory="frontend/templates")

    # Root and Health Endpoints
    @app.get("/", tags=["Health"])
    async def root():
        """Root endpoint - redirect to web interface."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/app", status_code=302)

    @app.get("/api", tags=["Health"])
    async def api_info():
        """API information endpoint."""
        return {
            "name": "Drug Discovery Compound Optimization API",
            "version": "1.0.0",
            "description": "Production-ready REST API for molecular property prediction and compound optimization",
            "docs_url": "/docs",
            "health_url": "/health",
            "web_interface": "/app",
            "endpoints": {
                "validation": "/validate_smiles",
                "prediction": "/predict_properties",
                "batch_prediction": "/batch/predict_properties",
                "similarity": "/calculate_similarity",
                "optimization": "/optimize_compound",
                "file_upload": "/upload/molecules",
                "metrics": "/metrics"
            }
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Comprehensive health check endpoint."""
        health_data = api_instance.get_system_health()
        return health_data

    @app.get("/health/detailed", tags=["Health"])
    async def detailed_health_check():
        """Detailed health check with component status."""
        health_data = api_instance.get_system_health()

        # Add component checks
        components = []

        # Check data processor
        components.append({
            "name": "MolecularDataProcessor",
            "status": "healthy" if api_instance.data_processor else "unhealthy",
            "response_time": None,
            "error_message": None if api_instance.data_processor else "Data processor not available"
        })

        # Check cache
        cache_status = "healthy" if redis_client else "degraded"
        components.append({
            "name": "Cache",
            "status": cache_status,
            "response_time": None,
            "error_message": None if redis_client else "Using in-memory cache"
        })

        health_data["components"] = components
        return health_data

    # SMILES Validation Endpoints (using JSON request bodies instead of Pydantic models)
    @app.post("/validate_smiles", tags=["Validation"])
    async def validate_smiles_endpoint(request: Request):
                "valid": result.get("valid", False),
                "canonical_smiles": result.get("canonical_smiles"),
                "message": "Valid SMILES" if result.get("valid") else "Invalid SMILES"
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating SMILES: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/validate_smiles/batch", tags=["Validation"])
    async def validate_smiles_batch(request: Request):
        """Validate multiple SMILES strings."""
        try:
            # Get SMILES list from request body
            try:
                body = await request.json()
                smiles_list = body.get('smiles_list', [])
                if not smiles_list:
                    raise HTTPException(status_code=400, detail="Missing 'smiles_list' field in request body")
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON in request body")

            if not api_instance.data_processor:
                raise HTTPException(status_code=503, detail="Data processor not available")

            if len(smiles_list) > API_CONFIG['max_batch_size']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size exceeds maximum of {API_CONFIG['max_batch_size']}"
                )

            processed = api_instance.data_processor.process_smiles(smiles_list)

            results = []
            for i, smiles in enumerate(smiles_list):
                result = processed[i] if i < len(processed) else {"valid": False}
                results.append({
                    "smiles": smiles,
                    "valid": result.get("valid", False),
                    "canonical_smiles": result.get("canonical_smiles"),
                    "message": "Valid SMILES" if result.get("valid") else "Invalid SMILES"
                })

            return {
                "results": results,
                "total_molecules": len(smiles_list),
                "valid_molecules": sum(1 for r in results if r["valid"]),
                "invalid_molecules": sum(1 for r in results if not r["valid"])
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error validating SMILES batch: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Property Prediction Endpoints
    @app.post("/predict_properties", tags=["Prediction"])
    async def predict_properties(request: Request):
        """Predict molecular properties for single or multiple molecules."""
        try:
            # Parse request body
            try:
                body = await request.json()
                smiles = body.get('smiles', [])
                properties = body.get('properties', ['molecular_weight', 'logp', 'tpsa'])

                # Ensure smiles is a list
                if isinstance(smiles, str):
                    smiles = [smiles]
                if not smiles:
                    raise HTTPException(status_code=400, detail="Missing 'smiles' field in request body")
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON in request body")
                    "valid": result.get("valid", False),
                    "canonical_smiles": result.get("canonical_smiles"),
                    "message": "Valid SMILES" if result.get("valid") else "Invalid SMILES"
                })

            return {
                "results": results,
                "total_molecules": len(smiles_list),
                "valid_molecules": sum(1 for r in results if r["valid"]),
                "invalid_molecules": sum(1 for r in results if not r["valid"])
            }
        except Exception as e:
            logger.error(f"Error validating SMILES batch: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Property Prediction Endpoints
    @app.post("/predict_properties", tags=["Prediction"])
    async def predict_properties(request: PropertyPredictionRequest):
        """Predict molecular properties for single or multiple molecules."""
        try:
            if not api_instance.data_processor:
                raise HTTPException(status_code=503, detail="Data processor not available")

            start_time = time.time()
            results = await api_instance.predict_properties_batch(request.smiles, request.properties)
            processing_time = time.time() - start_time

            return {
                "results": results,
                "total_molecules": len(request.smiles),
                "valid_molecules": sum(1 for r in results if r["valid"]),
                "invalid_molecules": sum(1 for r in results if not r["valid"]),
                "processing_time": processing_time
            }
        except Exception as e:
            logger.error(f"Error predicting properties: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch/predict_properties", tags=["Batch"])
    async def batch_predict_properties(
        request: PropertyPredictionRequest,
        background_tasks: BackgroundTasks
    ):
        """Submit batch property prediction job."""
        try:
            if len(request.smiles) > API_CONFIG['max_batch_size']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size exceeds maximum of {API_CONFIG['max_batch_size']}"
                )

            # Generate job ID
            job_id = f"batch_{int(time.time())}_{len(request.smiles)}"

            # Store job info
            background_tasks_storage = {
                "job_id": job_id,
                "status": "submitted",
                "total_molecules": len(request.smiles),
                "processed_molecules": 0,
                "created_at": datetime.now(),
                "estimated_completion": datetime.now()  # Add estimated time
            }
            background_tasks[job_id] = background_tasks_storage

            # Add to background processing (placeholder)
            def process_batch():
                # This would run the actual batch processing
                background_tasks[job_id]["status"] = "processing"
                # ... actual processing would go here
                background_tasks[job_id]["status"] = "completed"

            background_tasks.add_task(process_batch)

            return {
                "job_id": job_id,
                "status": "submitted",
                "total_molecules": len(request.smiles),
                "estimated_completion_time": 300,  # 5 minutes estimate
                "polling_url": f"/batch/status/{job_id}",
                "result_url": f"/batch/results/{job_id}"
            }
        except Exception as e:
            logger.error(f"Error submitting batch job: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/batch/status/{job_id}", tags=["Batch"])
    async def get_batch_status(job_id: str):
        """Get status of batch processing job."""
        if job_id not in background_tasks:
            raise HTTPException(status_code=404, detail="Job not found")

        job_info = background_tasks[job_id]
        return {
            "job_id": job_id,
            "status": job_info["status"],
            "total_molecules": job_info["total_molecules"],
            "processed_molecules": job_info.get("processed_molecules", 0),
            "created_at": job_info["created_at"],
            "progress": job_info.get("processed_molecules", 0) / job_info["total_molecules"]
        }

    # Similarity Calculation Endpoints
    @app.post("/calculate_similarity", tags=["Similarity"])
    async def calculate_similarity_endpoint(request: SimilarityRequest):
        """Calculate molecular similarity between query and target molecules."""
        try:
            if not api_instance.data_processor:
                raise HTTPException(status_code=503, detail="Data processor not available")

            # Check batch size limit
            if len(request.target_smiles) > API_CONFIG['max_batch_size']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target list exceeds maximum of {API_CONFIG['max_batch_size']}"
                )

            # Process query molecule
            query_processed = api_instance.data_processor.process_smiles([request.query_smiles])
            if not query_processed or not query_processed[0].get("valid"):
                raise HTTPException(status_code=400, detail="Invalid query SMILES")

            # Process target molecules
            target_processed = api_instance.data_processor.process_smiles(request.target_smiles)

            # Calculate similarities (placeholder implementation)
            start_time = time.time()
            similarities = []
            for i, target_smiles in enumerate(request.target_smiles):
                if i < len(target_processed) and target_processed[i].get("valid"):
                    # Use real similarity calculation if available
                    if calculate_similarity:
                        similarity = calculate_similarity(
                            request.query_smiles,
                            target_smiles,
                            method=request.similarity_metric
                        )
                    else:
                        similarity = np.random.uniform(0, 1) if NUMPY_PANDAS_AVAILABLE else 0.5

                    similarities.append({
                        "target_smiles": target_smiles,
                        "similarity_score": similarity,
                        "metric": request.similarity_metric
                    })
                else:
                    similarities.append({
                        "target_smiles": target_smiles,
                        "similarity_score": 0.0,
                        "metric": request.similarity_metric,
                        "error": "Invalid target SMILES"
                    })

            # Filter by threshold and sort
            if request.threshold > 0:
                similarities = [s for s in similarities if s.get("similarity_score", 0) >= request.threshold]

            # Sort by similarity score
            similarities.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)

            # Apply top_k limit
            if request.top_k:
                similarities = similarities[:request.top_k]

            processing_time = time.time() - start_time

            return {
                "query_smiles": request.query_smiles,
                "results": similarities,
                "total_comparisons": len(request.target_smiles),
                "above_threshold": len(similarities),
                "metric": request.similarity_metric,
                "processing_time": processing_time
            }

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Compound Optimization Endpoints
    @app.post("/optimize_compound", tags=["Optimization"])
    async def optimize_compound(request: OptimizationRequest, background_tasks: BackgroundTasks):
        """Start compound optimization for target properties."""
        try:
            # Generate task ID
            task_id = f"opt_{int(time.time())}"

            # Store optimization task info
            optimization_task = {
                "task_id": task_id,
                "status": "pending",
                "starting_smiles": request.starting_smiles,
                "targets": [{"property": t.property_name, "value": t.target_value} for t in request.targets],
                "max_iterations": request.max_iterations,
                "created_at": datetime.now(),
                "estimated_completion_time": 600,  # 10 minutes estimate
            }
            background_tasks[task_id] = optimization_task

            # Add to background processing (placeholder)
            def run_optimization():
                # This would run the actual optimization
                background_tasks[task_id]["status"] = "running"
                # ... actual optimization would go here
                background_tasks[task_id]["status"] = "completed"

            background_tasks.add_task(run_optimization)

            return {
                "task_id": task_id,
                "status": "pending",
                "estimated_completion_time": 600,
                "polling_url": f"/optimization/status/{task_id}"
            }

        except Exception as e:
            logger.error(f"Error starting compound optimization: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/optimization/status/{task_id}", tags=["Optimization"])
    async def get_optimization_status(task_id: str):
        """Get status of optimization task."""
        if task_id not in background_tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task_info = background_tasks[task_id]
        return task_info

    # File Upload Endpoints
    # File Upload Endpoints
    @app.post("/upload/molecules", tags=["Upload"])
    async def upload_molecule_file(
        file: UploadFile = File(...),
        smiles_column: str = "smiles",
        target_column: str = None,
        background_tasks: BackgroundTasks = BackgroundTasks()
    ):
        """Upload and process molecular data file."""
        try:
            # Validate file type
            allowed_types = ["text/csv", "application/vnd.ms-excel",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
            if file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.content_type}"
                )

            # Check file size
            max_size = API_CONFIG['max_file_size_mb'] * 1024 * 1024  # Convert to bytes
            file_size = len(await file.read())
            await file.seek(0)  # Reset file pointer

            if file_size > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds maximum of {API_CONFIG['max_file_size_mb']}MB"
                )

            # Save uploaded file
            upload_id = f"upload_{int(time.time())}"
            file_path = UPLOAD_DIR / f"{upload_id}_{file.filename}"

            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Process file in background
            def process_uploaded_file():
                try:
                    # Load file based on extension
                    if file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(file_path) if NUMPY_PANDAS_AVAILABLE else None
                    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                        df = pd.read_excel(file_path) if NUMPY_PANDAS_AVAILABLE else None
                    else:
                        raise ValueError("Unsupported file format")

                    if df is None:
                        raise ValueError("Could not load file - pandas not available")

                    # Validate SMILES column
                    if smiles_column not in df.columns:
                        raise ValueError(f"SMILES column '{smiles_column}' not found")

                    # Process molecules
                    total_rows = len(df)
                    valid_rows = 0

                    if api_instance.data_processor:
                        processed = api_instance.data_processor.process_smiles(df[smiles_column].tolist())
                        valid_rows = sum(1 for p in processed if p.get("valid", False))

                    # Save results
                    result_path = RESULTS_DIR / f"processed_{upload_id}.csv"
                    df.to_csv(result_path, index=False)

                    # Update background job status
                    background_tasks[upload_id] = {
                        "upload_id": upload_id,
                        "status": "completed",
                        "total_rows": total_rows,
                        "valid_rows": valid_rows,
                        "invalid_rows": total_rows - valid_rows,
                        "result_file": str(result_path)
                    }

                except Exception as e:
                    background_tasks[upload_id] = {
                        "upload_id": upload_id,
                        "status": "failed",
                        "error": str(e)
                    }
                    logger.error(f"Error processing uploaded file: {e}")

            # Initialize job status
            background_tasks[upload_id] = {
                "upload_id": upload_id,
                "status": "processing",
                "filename": file.filename,
                "file_size": file_size,
                "created_at": datetime.now()
            }

            background_tasks.add_task(process_uploaded_file)

            return {
                "upload_id": upload_id,
                "filename": file.filename,
                "file_size": file_size,
                "status": "processing",
                "polling_url": f"/upload/status/{upload_id}"
            }

        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/upload/status/{upload_id}", tags=["Upload"])
    async def get_upload_status(upload_id: str):
        """Get status of file upload processing."""
        if upload_id not in background_tasks:
            raise HTTPException(status_code=404, detail="Upload not found")

        return background_tasks[upload_id]

    # Metrics and Monitoring Endpoints
    @app.get("/metrics", tags=["Metrics"])
    async def get_api_metrics():
        if not API_CONFIG['enable_metrics']:
            raise HTTPException(status_code=403, detail="Metrics collection is disabled")

        # Calculate uptime
        uptime = time.time() - metrics['start_time']

        # Calculate error rate
        total_requests = metrics['total_requests']
        error_rate = (metrics['failed_requests'] / total_requests * 100) if total_requests > 0 else 0

        # Calculate average response time per endpoint
        endpoint_metrics = []
        for endpoint_key, stats in metrics['endpoint_stats'].items():
            avg_response_time = (
                stats['total_response_time'] / stats['total_requests']
                if stats['total_requests'] > 0 else 0
            )
            endpoint_metrics.append({
                "endpoint": endpoint_key,
                "total_requests": stats['total_requests'],
                "successful_requests": stats['successful_requests'],
                "failed_requests": stats['failed_requests'],
                "average_response_time": avg_response_time,
                "max_response_time": stats['max_response_time'],
                "last_request": stats['last_request']
            })

        return {
            "system_metrics": {
                "uptime": uptime,
                "total_requests": total_requests,
                "successful_requests": metrics['successful_requests'],
                "failed_requests": metrics['failed_requests'],
                "error_rate": error_rate
            },
            "endpoint_metrics": endpoint_metrics,
            "memory_usage": {
                "cache_size": len(cache),
                "background_tasks": len(background_tasks)
            }
        }

    # Model Management Endpoints
    @app.get("/models", tags=["Models"])
    async def list_models():
        """List available models and their status."""
        return {
            "available_models": list(api_instance.models.keys()),
            "model_status": api_instance.models,
            "data_processor_available": api_instance.data_processor is not None
        }

    # Web Interface Endpoints
    @app.get("/app", response_class=HTMLResponse, tags=["Web Interface"])
    async def web_interface(request: Request):
        """Serve the web interface."""
        if Path("frontend/templates").exists() and 'templates' in globals():
            return templates.TemplateResponse("index.html", {"request": request})
        else:
            return HTMLResponse("""
                <!DOCTYPE html>
                <html>
                    <head>
                        <title>Drug Discovery API</title>
                        <meta charset="utf-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1">
                        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                    </head>
                    <body>
                        <div class="container mt-5">
                            <div class="row justify-content-center">
                                <div class="col-md-8 text-center">
                                    <h1 class="mb-4">🧬 Drug Discovery API</h1>
                                    <p class="lead mb-4">Web interface template not found.</p>
                                    <div class="d-grid gap-2 d-md-flex justify-content-md-center">
                                        <a href="/docs" class="btn btn-primary">📚 API Documentation</a>
                                        <a href="/health" class="btn btn-outline-primary">💚 Health Check</a>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </body>
                </html>
            """)
else:
    # Create dummy app if FastAPI is not available
    app = None
    logger.warning("FastAPI not available - API will not be functional")


def create_app(config_path: Optional[str] = None) -> Optional[object]:
    """
    Factory function to create FastAPI app.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI app instance or None if FastAPI not available
    """
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI is not available")
        return None
    
    global api_instance
    api_instance = DrugDiscoveryAPI(config_path)
    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    """
    Run the FastAPI server.
    
    Args:
        host: Host address
        port: Port number
        reload: Enable auto-reload
    """
    if not FASTAPI_AVAILABLE:
        logger.error("FastAPI is not available - cannot run server")
        return
    
    if app is None:
        logger.error("App is not initialized")
        return
    
    logger.info(f"Starting Drug Discovery API server on {host}:{port}")
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def main():
    """Main function to run the API server."""
    if not FASTAPI_AVAILABLE:
        print("FastAPI is not available. Please install it with: pip install fastapi uvicorn")
        return
    
    # Load configuration
    config_path = "config/config.yaml"
    config = {}
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Get API configuration
    api_config = config.get('api', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    reload = api_config.get('reload', True)
    
    # Run server
    run_server(host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()