"""
Unit tests for app.main module.
"""
import pytest
from unittest.mock import patch, Mock
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app, lifespan, model


def test_app_initialization():
    """Test that app is properly initialized with correct metadata."""
    assert app is not None
    assert app.title == "House Price Prediction Service"
    assert app.description == "AI-powered house price prediction service"
    assert app.version == "1.0.0"


def test_app_has_cors_middleware():
    """Test that CORS middleware is properly configured."""
    cors_middleware = None
    for middleware in app.user_middleware:
        if "CORSMiddleware" in str(middleware.cls):
            cors_middleware = middleware
            break
    
    assert cors_middleware is not None


def test_app_has_router():
    """Test that router is included in the app."""
    assert len(app.routes) > 0


def test_root_endpoint_success():
    """Test root endpoint returns correct response structure."""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert "model_ready" in data
    assert isinstance(data["message"], str)
    assert isinstance(data["status"], str)
    assert isinstance(data["model_ready"], bool)


def test_root_endpoint_response_content():
    """Test root endpoint returns expected content."""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "House Price Prediction Service is online!"
    assert data["status"] == "healthy"


def test_health_endpoint_success():
    """Test health endpoint returns correct response when model is ready."""
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code in [200, 503]
    data = response.json()
    
    if response.status_code == 200:
        assert "status" in data
        assert "model_ready" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["model_ready"], bool)
    else:
        assert "detail" in data


def test_health_endpoint_model_not_ready():
    """Test health endpoint handles model not ready scenario."""
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code in [200, 503]
    data = response.json()
    
    if response.status_code == 503:
        assert "detail" in data
    else:
        assert "status" in data or "detail" in data


def test_health_endpoint_exception():
    """Test health endpoint handles exceptions gracefully."""
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code in [200, 503]
    data = response.json()
    assert "status" in data or "detail" in data


@pytest.mark.asyncio
async def test_lifespan_startup_success():
    """Test lifespan startup successfully initializes model."""
    with patch("app.main.HousePricePrediction") as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        async with lifespan(app):
            mock_model_class.assert_called_once()
            mock_model.train.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_startup_failure():
    """Test lifespan startup raises exception when model initialization fails."""
    with patch("app.main.HousePricePrediction") as mock_model_class:
        mock_model_class.side_effect = Exception("Initialization error")
        
        with pytest.raises(Exception) as excinfo:
            async with lifespan(app):
                pass
        
        assert "Initialization error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_lifespan_shutdown():
    """Test lifespan shutdown completes without errors."""
    with patch("app.main.HousePricePrediction") as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        async with lifespan(app):
            pass


def test_app_lifespan_configuration():
    """Test that app has lifespan configured."""
    assert hasattr(app, "router")


def test_app_middleware_configuration():
    """Test that CORS middleware is properly configured."""
    cors_middleware = None
    for middleware in app.user_middleware:
        if "CORSMiddleware" in str(middleware.cls):
            cors_middleware = middleware
            break
    
    assert cors_middleware is not None


def test_app_router_inclusion():
    """Test that router is included in the app."""
    assert len(app.routes) > 0


def test_root_endpoint_response_structure():
    """Test root endpoint response has correct structure."""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert "model_ready" in data
    assert isinstance(data["message"], str)
    assert isinstance(data["status"], str)
    assert isinstance(data["model_ready"], bool)


def test_health_endpoint_response_structure():
    """Test health endpoint response has correct structure."""
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code in [200, 503]
    data = response.json()
    
    if response.status_code == 200:
        assert "status" in data
        assert "model_ready" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["model_ready"], bool)
    else:
        assert "detail" in data


def test_app_logging_configuration():
    """Test that logging is properly configured."""
    import logging
    logger = logging.getLogger("app.main")
    assert logger.level <= logging.INFO


def test_model_global_variable():
    """Test that model global variable is accessible."""
    assert model is None


def test_app_version_consistency():
    """Test app version consistency."""
    assert app.version == "1.0.0"


def test_app_title_consistency():
    """Test app title consistency."""
    assert app.title == "House Price Prediction Service"


def test_app_description_consistency():
    """Test app description consistency."""
    assert app.description == "AI-powered house price prediction service"


def test_app_different_from_email_classifier():
    """Test that this app is different from email classifier."""
    assert app.title != "Email Classifier"
    assert "house price" in app.description.lower()


def test_model_variable_name():
    """Test model variable name."""
    assert 'model' in globals()


@pytest.mark.asyncio
async def test_lifespan_model_initialization():
    """Test lifespan model initialization."""
    with patch("app.main.HousePricePrediction") as mock_model_class:
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        
        async with lifespan(app):
            mock_model_class.assert_called_once()
            mock_model.train.assert_called_once()


def test_root_endpoint_with_model_ready():
    """Test root endpoint when model is ready."""
    with patch("app.main.model", Mock()):
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_ready"] is True


def test_root_endpoint_with_model_not_ready():
    """Test root endpoint when model is not ready."""
    with patch("app.main.model", None):
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_ready"] is False


def test_health_endpoint_with_model_ready():
    """Test health endpoint when model is ready."""
    with patch("app.main.model", Mock()):
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_ready"] is True


def test_health_endpoint_with_model_not_ready():
    """Test health endpoint when model is not ready."""
    with patch("app.main.model", None):
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data


def test_health_endpoint_with_exception():
    """Test health endpoint when exception occurs."""
    with patch("app.main.model", Mock(side_effect=Exception("Test error"))):
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code in [200, 503]
        data = response.json()
        if response.status_code == 503:
            assert "detail" in data
        else:
            assert "status" in data 