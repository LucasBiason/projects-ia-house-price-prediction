"""
Unit tests for app.views module.
"""
import pytest
from unittest.mock import patch
from fastapi import HTTPException

from app.views import router, predict
from app.schemas import HouseFeatures


def test_router_initialization():
    """Test that router is properly initialized."""
    assert router is not None
    assert hasattr(router, "routes")


def test_predict_success():
    """Test predict function with valid features returns correct prediction."""
    features = HouseFeatures(
        tamanho=150.5,
        localizacao="São Paulo",
        quantidade_quartos=3
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 500000.0
        
        result = predict(features)
        
        assert result.price == 500000.0
        mock_model.predict.assert_called_once_with([150.5, "São Paulo", 3])


def test_predict_file_not_found_error():
    """Test predict function raises HTTPException when model file not found."""
    features = HouseFeatures(
        tamanho=150.5,
        localizacao="São Paulo",
        quantidade_quartos=3
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.side_effect = FileNotFoundError("Model file not found")
        
        with pytest.raises(HTTPException) as excinfo:
            predict(features)
        
        assert excinfo.value.status_code == 500
        assert "Model file not found" in str(excinfo.value.detail)


def test_predict_generic_exception():
    """Test predict function raises exception when model fails."""
    features = HouseFeatures(
        tamanho=150.5,
        localizacao="São Paulo",
        quantidade_quartos=3
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.side_effect = Exception("Generic error")
        
        with pytest.raises(Exception) as excinfo:
            predict(features)
        
        assert "Generic error" in str(excinfo.value)


def test_predict_with_different_feature_values():
    """Test predict function with different feature values."""
    features = HouseFeatures(
        tamanho=200.0,
        localizacao="Rio de Janeiro",
        quantidade_quartos=4
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 750000.0
        
        result = predict(features)
        
        assert result.price == 750000.0
        mock_model.predict.assert_called_once_with([200.0, "Rio de Janeiro", 4])


def test_predict_with_zero_values():
    """Test predict function with zero values."""
    features = HouseFeatures(
        tamanho=0.0,
        localizacao="Test",
        quantidade_quartos=0
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 0.0
        
        result = predict(features)
        
        assert result.price == 0.0
        mock_model.predict.assert_called_once_with([0.0, "Test", 0])


def test_predict_with_negative_values():
    """Test predict function with negative values."""
    features = HouseFeatures(
        tamanho=-50.0,
        localizacao="Test",
        quantidade_quartos=-1
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = -1000.0
        
        result = predict(features)
        
        assert result.price == -1000.0
        mock_model.predict.assert_called_once_with([-50.0, "Test", -1])


def test_predict_with_large_values():
    """Test predict function with large values."""
    features = HouseFeatures(
        tamanho=999999.99,
        localizacao="Large City",
        quantidade_quartos=999
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 999999999.99
        
        result = predict(features)
        
        assert result.price == 999999999.99
        mock_model.predict.assert_called_once_with([999999.99, "Large City", 999])


def test_predict_with_decimal_values():
    """Test predict function with decimal values."""
    features = HouseFeatures(
        tamanho=150.75,
        localizacao="São Paulo",
        quantidade_quartos=3
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 500123.45
        
        result = predict(features)
        
        assert result.price == 500123.45
        mock_model.predict.assert_called_once_with([150.75, "São Paulo", 3])


def test_predict_with_empty_string_location():
    """Test predict function with empty string location."""
    features = HouseFeatures(
        tamanho=150.5,
        localizacao="",
        quantidade_quartos=3
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 400000.0
        
        result = predict(features)
        
        assert result.price == 400000.0
        mock_model.predict.assert_called_once_with([150.5, "", 3])


def test_predict_with_special_characters_location():
    """Test predict function with special characters in location."""
    features = HouseFeatures(
        tamanho=150.5,
        localizacao="São Paulo-SP",
        quantidade_quartos=3
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 550000.0
        
        result = predict(features)
        
        assert result.price == 550000.0
        mock_model.predict.assert_called_once_with([150.5, "São Paulo-SP", 3])


def test_predict_with_very_small_values():
    """Test predict function with very small values."""
    features = HouseFeatures(
        tamanho=0.1,
        localizacao="Test",
        quantidade_quartos=1
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 100000.0
        
        result = predict(features)
        
        assert result.price == 100000.0
        mock_model.predict.assert_called_once_with([0.1, "Test", 1])


def test_predict_with_very_large_room_count():
    """Test predict function with very large room count."""
    features = HouseFeatures(
        tamanho=500.0,
        localizacao="Luxury Area",
        quantidade_quartos=20
    )
    
    with patch("app.views.model") as mock_model:
        mock_model.predict.return_value = 2000000.0
        
        result = predict(features)
        
        assert result.price == 2000000.0
        mock_model.predict.assert_called_once_with([500.0, "Luxury Area", 20]) 