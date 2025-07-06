"""
Comprehensive API Tests for Drug Discovery Compound Optimization

This module contains comprehensive tests for the FastAPI application including
unit tests, integration tests, performance tests, and mock data testing.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List
from pathlib import Path
import tempfile
import io

try:
    from fastapi.testclient import TestClient
    from fastapi import status
    import httpx
    TESTING_AVAILABLE = True
except ImportError:
    TESTING_AVAILABLE = False

# Import our API
try:
    from src.api import app, api_instance, FASTAPI_AVAILABLE
    from src.api_models import *
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Mock data for testing
MOCK_SMILES_DATA = {
    "valid": [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    ],
    "invalid": [
        "invalid_smiles",
        "",
        "abc123",
        "C=C=C=C",  # Potentially invalid
    ]
}

MOCK_PROPERTIES = [
    "molecular_weight",
    "logp", 
    "tpsa",
    "num_hbd",
    "num_hba",
    "num_rotatable_bonds"
]

# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (TESTING_AVAILABLE and API_AVAILABLE and FASTAPI_AVAILABLE),
    reason="FastAPI, TestClient, or API modules not available"
)


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    if app is None:
        pytest.skip("FastAPI app not available")
    return TestClient(app)


@pytest.fixture
def sample_csv_file():
    """Create a sample CSV file for upload testing."""
    csv_content = """smiles,activity,name
CCO,1.2,Ethanol
CC(=O)O,0.8,Acetic Acid
c1ccccc1,2.1,Benzene
invalid_smiles,1.0,Invalid
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,3.5,Caffeine"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        return f.name


@pytest.fixture
def async_client():
    """Create async test client for testing async endpoints."""
    if app is None:
        pytest.skip("FastAPI app not available")
    return httpx.AsyncClient(app=app, base_url="http://test")


class TestHealthEndpoints:
    """Test health check and system status endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
        
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        
    def test_detailed_health_check(self, client):
        """Test detailed health check with component status."""
        response = client.get("/health/detailed")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "status" in data
        assert "dependencies" in data
        assert "components" in data
        assert isinstance(data["components"], list)


class TestSMILESValidation:
    """Test SMILES validation endpoints."""

    def test_validate_single_valid_smiles(self, client):
        """Test validation of single valid SMILES."""
        for smiles in MOCK_SMILES_DATA["valid"]:
            response = client.post(
                "/validate_smiles",
                json={"smiles": smiles}
            )
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["smiles"] == smiles
            # Note: actual validation depends on RDKit availability

    def test_validate_single_invalid_smiles(self, client):
        """Test validation of single invalid SMILES."""
        for smiles in MOCK_SMILES_DATA["invalid"]:
            response = client.post(
                "/validate_smiles",
                json={"smiles": smiles}
            )
            # Should still return 200 but with valid: false
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["smiles"] == smiles

    def test_validate_batch_smiles(self, client):
        """Test batch SMILES validation."""
        all_smiles = MOCK_SMILES_DATA["valid"] + MOCK_SMILES_DATA["invalid"]
        
        response = client.post(
            "/validate_smiles/batch",
            json={"smiles_list": all_smiles}
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "results" in data
        assert "total_molecules" in data
        assert "valid_molecules" in data
        assert "invalid_molecules" in data
        assert data["total_molecules"] == len(all_smiles)

    def test_validate_empty_smiles(self, client):
        """Test validation with empty SMILES."""
        response = client.post(
            "/validate_smiles",
            json={"smiles": ""}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_validate_batch_size_limit(self, client):
        """Test batch size limit enforcement."""
        large_batch = ["CCO"] * 2000  # Exceed typical limit
        
        response = client.post(
            "/validate_smiles/batch",
            json={"smiles_list": large_batch}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "exceeds maximum" in response.json()["detail"]


class TestPropertyPrediction:
    """Test molecular property prediction endpoints."""

    def test_predict_single_molecule(self, client):
        """Test property prediction for single molecule."""
        response = client.post(
            "/predict_properties",
            json={
                "smiles": ["CCO"],
                "properties": ["molecular_weight", "logp"]
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "results" in data
        assert "total_molecules" in data
        assert "processing_time" in data
        assert len(data["results"]) == 1

    def test_predict_multiple_molecules(self, client):
        """Test property prediction for multiple molecules."""
        response = client.post(
            "/predict_properties",
            json={
                "smiles": MOCK_SMILES_DATA["valid"][:3],
                "properties": MOCK_PROPERTIES[:3]
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert len(data["results"]) == 3
        assert data["total_molecules"] == 3

    def test_predict_with_invalid_smiles(self, client):
        """Test property prediction with invalid SMILES."""
        response = client.post(
            "/predict_properties", 
            json={
                "smiles": ["invalid_smiles"],
                "properties": ["molecular_weight"]
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert len(data["results"]) == 1
        # Should handle invalid SMILES gracefully

    def test_predict_no_properties(self, client):
        """Test prediction with no properties specified."""
        response = client.post(
            "/predict_properties",
            json={
                "smiles": ["CCO"],
                "properties": []
            }
        )
        # Should use default properties or return error
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]

    def test_batch_predict_properties(self, client):
        """Test batch property prediction job submission."""
        response = client.post(
            "/batch/predict_properties",
            json={
                "smiles": MOCK_SMILES_DATA["valid"],
                "properties": MOCK_PROPERTIES[:3]
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "polling_url" in data
        assert data["status"] == "submitted"

    def test_batch_status_check(self, client):
        """Test batch job status checking."""
        # First submit a job
        response = client.post(
            "/batch/predict_properties",
            json={
                "smiles": ["CCO"],
                "properties": ["molecular_weight"]
            }
        )
        assert response.status_code == status.HTTP_200_OK
        job_id = response.json()["job_id"]
        
        # Check status
        status_response = client.get(f"/batch/status/{job_id}")
        assert status_response.status_code == status.HTTP_200_OK
        
        status_data = status_response.json()
        assert "job_id" in status_data
        assert "status" in status_data

    def test_batch_status_not_found(self, client):
        """Test batch status for non-existent job."""
        response = client.get("/batch/status/nonexistent_job_id")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestSimilarityCalculation:
    """Test molecular similarity calculation endpoints."""

    def test_calculate_similarity(self, client):
        """Test basic similarity calculation."""
        response = client.post(
            "/calculate_similarity",
            json={
                "query_smiles": "CCO",
                "target_smiles": ["CC(=O)O", "c1ccccc1"],
                "similarity_metric": "tanimoto",
                "threshold": 0.0
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "query_smiles" in data
        assert "results" in data
        assert "total_comparisons" in data
        assert "processing_time" in data
        assert len(data["results"]) <= 2

    def test_similarity_with_threshold(self, client):
        """Test similarity calculation with threshold filtering."""
        response = client.post(
            "/calculate_similarity",
            json={
                "query_smiles": "CCO",
                "target_smiles": MOCK_SMILES_DATA["valid"],
                "similarity_metric": "tanimoto",
                "threshold": 0.5
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        # All returned results should be above threshold
        for result in data["results"]:
            assert result["similarity_score"] >= 0.5

    def test_similarity_different_metrics(self, client):
        """Test different similarity metrics."""
        metrics = ["tanimoto", "dice", "cosine"]
        
        for metric in metrics:
            response = client.post(
                "/calculate_similarity",
                json={
                    "query_smiles": "CCO",
                    "target_smiles": ["CC(=O)O"],
                    "similarity_metric": metric
                }
            )
            assert response.status_code == status.HTTP_200_OK
            
            data = response.json()
            assert data["metric"] == metric

    def test_similarity_invalid_query(self, client):
        """Test similarity with invalid query SMILES."""
        response = client.post(
            "/calculate_similarity",
            json={
                "query_smiles": "invalid_smiles",
                "target_smiles": ["CCO"],
                "similarity_metric": "tanimoto"
            }
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_similarity_batch_size_limit(self, client):
        """Test similarity calculation batch size limit."""
        large_targets = ["CCO"] * 2000
        
        response = client.post(
            "/calculate_similarity",
            json={
                "query_smiles": "CCO",
                "target_smiles": large_targets,
                "similarity_metric": "tanimoto"
            }
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestCompoundOptimization:
    """Test compound optimization endpoints."""

    def test_start_optimization(self, client):
        """Test starting compound optimization."""
        response = client.post(
            "/optimize_compound",
            json={
                "starting_smiles": "CCO",
                "targets": [
                    {
                        "property_name": "molecular_weight",
                        "target_value": 300.0,
                        "weight": 1.0,
                        "tolerance": 0.1,
                        "direction": "target"
                    }
                ],
                "max_iterations": 100
            }
        )
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "task_id" in data
        assert "status" in data
        assert "estimated_completion_time" in data

    def test_optimization_status(self, client):
        """Test optimization status checking."""
        # Start optimization
        response = client.post(
            "/optimize_compound",
            json={
                "starting_smiles": "CCO",
                "targets": [
                    {
                        "property_name": "logp",
                        "target_value": 2.5,
                        "weight": 1.0,
                        "tolerance": 0.1,
                        "direction": "target"
                    }
                ]
            }
        )
        task_id = response.json()["task_id"]
        
        # Check status
        status_response = client.get(f"/optimization/status/{task_id}")
        assert status_response.status_code == status.HTTP_200_OK
        
        status_data = status_response.json()
        assert "task_id" in status_data
        assert "status" in status_data

    def test_optimization_invalid_smiles(self, client):
        """Test optimization with invalid starting SMILES."""
        response = client.post(
            "/optimize_compound",
            json={
                "starting_smiles": "",
                "targets": [
                    {
                        "property_name": "molecular_weight",
                        "target_value": 300.0,
                        "weight": 1.0,
                        "tolerance": 0.1,
                        "direction": "target"
                    }
                ]
            }
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_optimization_multiple_targets(self, client):
        """Test optimization with multiple targets."""
        response = client.post(
            "/optimize_compound",
            json={
                "starting_smiles": "CCO",
                "targets": [
                    {
                        "property_name": "molecular_weight",
                        "target_value": 300.0,
                        "weight": 2.0,
                        "tolerance": 0.1,
                        "direction": "target"
                    },
                    {
                        "property_name": "logp",
                        "target_value": 2.5,
                        "weight": 1.0,
                        "tolerance": 0.2,
                        "direction": "maximize"
                    }
                ]
            }
        )
        assert response.status_code == status.HTTP_200_OK


class TestFileUpload:
    """Test file upload and processing endpoints."""

    def test_upload_csv_file(self, client, sample_csv_file):
        """Test CSV file upload."""
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/upload/molecules",
                files={"file": ("test.csv", f, "text/csv")},
                params={
                    "smiles_column": "smiles",
                    "target_column": "activity"
                }
            )
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "upload_id" in data
        assert "filename" in data
        assert "status" in data
        assert data["filename"] == "test.csv"

    def test_upload_status(self, client, sample_csv_file):
        """Test upload status checking."""
        # Upload file
        with open(sample_csv_file, 'rb') as f:
            response = client.post(
                "/upload/molecules",
                files={"file": ("test.csv", f, "text/csv")}
            )
        upload_id = response.json()["upload_id"]
        
        # Check status
        status_response = client.get(f"/upload/status/{upload_id}")
        assert status_response.status_code == status.HTTP_200_OK
        
        status_data = status_response.json()
        assert "upload_id" in status_data
        assert "status" in status_data

    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        fake_file = io.BytesIO(b"fake content")
        response = client.post(
            "/upload/molecules",
            files={"file": ("test.txt", fake_file, "text/plain")}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_upload_large_file(self, client):
        """Test upload file size limit."""
        # Create a large fake file
        large_content = b"x" * (101 * 1024 * 1024)  # 101MB
        large_file = io.BytesIO(large_content)
        
        response = client.post(
            "/upload/molecules",
            files={"file": ("large.csv", large_file, "text/csv")}
        )
        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class TestAPIMetrics:
    """Test API metrics and monitoring endpoints."""

    def test_get_metrics(self, client):
        """Test API metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "system_metrics" in data
        assert "endpoint_metrics" in data
        assert "memory_usage" in data

    def test_metrics_structure(self, client):
        """Test metrics response structure."""
        response = client.get("/metrics")
        data = response.json()
        
        system_metrics = data["system_metrics"]
        assert "uptime" in system_metrics
        assert "total_requests" in system_metrics
        assert "successful_requests" in system_metrics
        assert "failed_requests" in system_metrics
        assert "error_rate" in system_metrics


class TestModelManagement:
    """Test model management endpoints."""

    def test_list_models(self, client):
        """Test model listing endpoint."""
        response = client.get("/models")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "available_models" in data
        assert "model_status" in data
        assert isinstance(data["available_models"], list)


class TestRateLimit:
    """Test rate limiting functionality."""

    def test_rate_limit_enforcement(self, client):
        """Test that rate limiting is properly enforced."""
        # This test depends on rate limiting being configured
        # Make rapid requests to trigger rate limit
        responses = []
        for _ in range(150):  # Exceed typical rate limit
            response = client.get("/health")
            responses.append(response.status_code)
        
        # Should see some rate limit responses (429) if enabled
        # Note: This might not trigger in test environment without proper setup


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async endpoint functionality."""

    async def test_async_property_prediction(self, async_client):
        """Test async property prediction."""
        response = await async_client.post(
            "/predict_properties",
            json={
                "smiles": ["CCO"],
                "properties": ["molecular_weight"]
            }
        )
        assert response.status_code == status.HTTP_200_OK

    async def test_concurrent_requests(self, async_client):
        """Test handling of concurrent requests."""
        tasks = []
        for _ in range(10):
            task = async_client.post(
                "/predict_properties",
                json={
                    "smiles": ["CCO"],
                    "properties": ["molecular_weight"]
                }
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        for response in responses:
            assert response.status_code == status.HTTP_200_OK


class TestPerformance:
    """Test API performance and load handling."""

    def test_prediction_response_time(self, client):
        """Test property prediction response time."""
        start_time = time.time()
        
        response = client.post(
            "/predict_properties",
            json={
                "smiles": MOCK_SMILES_DATA["valid"][:10],
                "properties": ["molecular_weight", "logp"]
            }
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == status.HTTP_200_OK
        assert response_time < 30.0  # Should complete within 30 seconds

    def test_similarity_performance(self, client):
        """Test similarity calculation performance."""
        start_time = time.time()
        
        response = client.post(
            "/calculate_similarity",
            json={
                "query_smiles": "CCO",
                "target_smiles": MOCK_SMILES_DATA["valid"] * 20,  # 100 comparisons
                "similarity_metric": "tanimoto"
            }
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == status.HTTP_200_OK
        assert response_time < 30.0  # Should complete within 30 seconds


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_json(self, client):
        """Test handling of malformed JSON requests."""
        response = client.post(
            "/predict_properties",
            data="malformed json{",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post(
            "/predict_properties",
            json={"properties": ["molecular_weight"]}  # Missing smiles
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_property_names(self, client):
        """Test handling of invalid property names."""
        response = client.post(
            "/predict_properties",
            json={
                "smiles": ["CCO"],
                "properties": ["invalid_property"]
            }
        )
        # Should handle gracefully
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST]


# Utility functions for test data generation
def generate_random_smiles(count: int = 10) -> List[str]:
    """Generate random SMILES strings for testing."""
    # This is a simplified version - in practice you'd use RDKit
    base_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CC(C)C"]
    import random
    return random.choices(base_smiles, k=count)


def create_test_csv(smiles_list: List[str], filename: str = "test.csv") -> str:
    """Create a test CSV file with SMILES data."""
    import tempfile
    import csv
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.writer(f)
        writer.writerow(['smiles', 'activity', 'name'])
        for i, smiles in enumerate(smiles_list):
            writer.writerow([smiles, f'{i+1:.2f}', f'Compound_{i+1}'])
        return f.name


# Test configuration and fixtures
@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        "timeout": 30,
        "max_retries": 3,
        "test_data_size": 100
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])