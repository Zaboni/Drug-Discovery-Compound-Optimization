"""
FastAPI Web Service for Drug Discovery Compound Optimization

This module provides a REST API for molecular property prediction,
compound optimization, and drug discovery services.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import yaml

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, field_validator
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
    class uvicorn:
        @staticmethod
        def run(*args, **kwargs):
            pass
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

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
except ImportError:
    # Fallback for when modules aren't available
    MolecularDataProcessor = None
    PropertyPredictor = None
    RandomForestModel = None
    validate_smiles = None
    calculate_similarity = None

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
if FASTAPI_AVAILABLE:
    class SMILESInput(BaseModel):
        """Input model for SMILES strings."""
        smiles: str

    class PropertyPredictionRequest(BaseModel):
        """Request model for property prediction."""
        smiles: Union[str, List[str]]
        properties: List[str] = ["logp", "solubility", "molecular_weight"]

        @field_validator('smiles')
        @classmethod
        def validate_smiles_input(cls, v):
            if isinstance(v, str):
                return [v.strip()]
            elif isinstance(v, list):
                return [s.strip() for s in v if s and isinstance(s, str)]
            else:
                raise ValueError("SMILES must be a string or list of strings")

    class PropertyPredictionResponse(BaseModel):
        """Response model for property prediction."""
        smiles: str
        properties: Dict[str, float]
        valid: bool
        error: Optional[str] = None

    class SimilarityRequest(BaseModel):
        """Request model for molecular similarity calculation."""
        query_smiles: str
        target_smiles: List[str]
        similarity_metric: str = "tanimoto"

        @field_validator('similarity_metric')
        @classmethod
        def validate_metric(cls, v):
            allowed_metrics = ["tanimoto", "dice", "cosine"]
            if v not in allowed_metrics:
                raise ValueError(f"Similarity metric must be one of {allowed_metrics}")
            return v

    class OptimizationRequest(BaseModel):
        """Request model for compound optimization."""
        starting_smiles: str
        target_properties: Dict[str, float]
        optimization_steps: int = 100

        @field_validator('optimization_steps')
        @classmethod
        def validate_steps(cls, v):
            if v < 1 or v > 1000:
                raise ValueError("Optimization steps must be between 1 and 1000")
            return v

    class HealthResponse(BaseModel):
        """Health check response model."""
        status: str
        version: str
        models_loaded: Dict[str, bool]
        dependencies: Dict[str, bool]

else:
    # Dummy classes if FastAPI is not available
    class BaseModel:
        pass
    
    SMILESInput = PropertyPredictionRequest = PropertyPredictionResponse = BaseModel
    SimilarityRequest = OptimizationRequest = HealthResponse = BaseModel


class DrugDiscoveryAPI:
    """Main API class for drug discovery services."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Drug Discovery API.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.data_processor = None
        self.property_predictor = None
        self.models = {}
        
        # Initialize components
        self._initialize_components()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return {
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'reload': True
                },
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


# Create FastAPI app if available
if FASTAPI_AVAILABLE:
    # Initialize API
    api_instance = DrugDiscoveryAPI()
    
    app = FastAPI(
        title="Drug Discovery Compound Optimization API",
        description="REST API for molecular property prediction and compound optimization",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "Drug Discovery Compound Optimization API",
            "version": "1.0.0",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models_loaded={
                "property_predictor": bool(api_instance.models.get('property_predictor')),
                "data_processor": api_instance.data_processor is not None
            },
            dependencies={
                "fastapi": FASTAPI_AVAILABLE,
                "numpy_pandas": NUMPY_PANDAS_AVAILABLE,
                "rdkit": MolecularDataProcessor is not None
            }
        )
    
    @app.post("/validate_smiles")
    async def validate_smiles_endpoint(smiles_input: SMILESInput):
        """Validate SMILES string."""
        try:
            if not api_instance.data_processor:
                raise HTTPException(status_code=503, detail="Data processor not available")
            
            # Process SMILES
            processed = api_instance.data_processor.process_smiles([smiles_input.smiles])
            result = processed[0] if processed else {"valid": False}
            
            return {
                "smiles": smiles_input.smiles,
                "valid": result.get("valid", False),
                "canonical_smiles": result.get("canonical_smiles"),
                "message": "Valid SMILES" if result.get("valid") else "Invalid SMILES"
            }
        except Exception as e:
            logger.error(f"Error validating SMILES: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/predict_properties", response_model=List[PropertyPredictionResponse])
    async def predict_properties(request: PropertyPredictionRequest):
        """Predict molecular properties."""
        try:
            if not api_instance.data_processor:
                raise HTTPException(status_code=503, detail="Data processor not available")
            
            results = []
            
            for smiles in request.smiles:
                try:
                    # Process SMILES
                    processed = api_instance.data_processor.process_smiles([smiles])
                    
                    if not processed or not processed[0].get("valid"):
                        results.append(PropertyPredictionResponse(
                            smiles=smiles,
                            properties={},
                            valid=False,
                            error="Invalid SMILES"
                        ))
                        continue
                    
                    # Extract features
                    features_df = api_instance.data_processor.extract_features(processed)
                    
                    # Predict properties (placeholder implementation)
                    properties = {}
                    for prop in request.properties:
                        if prop == "molecular_weight":
                            properties[prop] = features_df.iloc[0].get("molecular_weight", 0.0)
                        elif prop == "logp":
                            properties[prop] = features_df.iloc[0].get("logp", 0.0)
                        elif prop == "tpsa":
                            properties[prop] = features_df.iloc[0].get("tpsa", 0.0)
                        else:
                            # Placeholder for other properties
                            properties[prop] = np.random.uniform(0, 1) if NUMPY_PANDAS_AVAILABLE else 0.5
                    
                    results.append(PropertyPredictionResponse(
                        smiles=smiles,
                        properties=properties,
                        valid=True
                    ))
                    
                except Exception as e:
                    results.append(PropertyPredictionResponse(
                        smiles=smiles,
                        properties={},
                        valid=False,
                        error=str(e)
                    ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting properties: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/calculate_similarity")
    async def calculate_similarity_endpoint(request: SimilarityRequest):
        """Calculate molecular similarity."""
        try:
            if not api_instance.data_processor:
                raise HTTPException(status_code=503, detail="Data processor not available")
            
            # Process query molecule
            query_processed = api_instance.data_processor.process_smiles([request.query_smiles])
            if not query_processed or not query_processed[0].get("valid"):
                raise HTTPException(status_code=400, detail="Invalid query SMILES")
            
            # Process target molecules
            target_processed = api_instance.data_processor.process_smiles(request.target_smiles)
            
            # Calculate similarities (placeholder implementation)
            similarities = []
            for i, target_smiles in enumerate(request.target_smiles):
                if i < len(target_processed) and target_processed[i].get("valid"):
                    # Placeholder similarity calculation
                    similarity = np.random.uniform(0, 1) if NUMPY_PANDAS_AVAILABLE else 0.5
                    similarities.append({
                        "target_smiles": target_smiles,
                        "similarity": similarity,
                        "metric": request.similarity_metric
                    })
                else:
                    similarities.append({
                        "target_smiles": target_smiles,
                        "similarity": 0.0,
                        "metric": request.similarity_metric,
                        "error": "Invalid target SMILES"
                    })
            
            return {
                "query_smiles": request.query_smiles,
                "similarities": similarities,
                "metric": request.similarity_metric
            }
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/optimize_compound")
    async def optimize_compound(request: OptimizationRequest, background_tasks: BackgroundTasks):
        """Optimize compound for target properties."""
        try:
            # This would typically be a long-running task
            # For now, return a placeholder response
            
            return {
                "message": "Optimization started",
                "starting_smiles": request.starting_smiles,
                "target_properties": request.target_properties,
                "optimization_steps": request.optimization_steps,
                "status": "running",
                "estimated_time": "5-10 minutes"
            }
            
        except Exception as e:
            logger.error(f"Error optimizing compound: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/models")
    async def list_models():
        """List available models."""
        return {
            "available_models": list(api_instance.models.keys()),
            "model_status": api_instance.models
        }

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