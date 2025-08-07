"""
Unit tests for app.schemas module.
"""
import pytest
from pydantic import ValidationError

from app.schemas import HouseFeatures, PricePrediction


def test_house_features_valid_data():
    """Test HouseFeatures with valid data returns correct values."""
    data = {
        "tamanho": 150.5,
        "localizacao": "São Paulo",
        "quantidade_quartos": 3
    }
    
    house_features = HouseFeatures(**data)
    
    assert house_features.tamanho == 150.5
    assert house_features.localizacao == "São Paulo"
    assert house_features.quantidade_quartos == 3


def test_house_features_invalid_tamanho():
    """Test HouseFeatures raises ValidationError when tamanho is invalid."""
    data = {
        "tamanho": "invalid",
        "localizacao": "São Paulo",
        "quantidade_quartos": 3
    }
    
    with pytest.raises(ValidationError) as excinfo:
        HouseFeatures(**data)
    
    assert "tamanho" in str(excinfo.value)


def test_house_features_invalid_quantidade_quartos():
    """Test HouseFeatures raises ValidationError when quantidade_quartos is invalid."""
    data = {
        "tamanho": 150.5,
        "localizacao": "São Paulo",
        "quantidade_quartos": "invalid"
    }
    
    with pytest.raises(ValidationError) as excinfo:
        HouseFeatures(**data)
    
    assert "quantidade_quartos" in str(excinfo.value)


def test_house_features_missing_required_field():
    """Test HouseFeatures raises ValidationError when required field is missing."""
    data = {
        "tamanho": 150.5,
        "localizacao": "São Paulo"
    }
    
    with pytest.raises(ValidationError) as excinfo:
        HouseFeatures(**data)
    
    assert "quantidade_quartos" in str(excinfo.value)


def test_house_features_extra_field():
    """Test HouseFeatures ignores extra fields."""
    data = {
        "tamanho": 150.5,
        "localizacao": "São Paulo",
        "quantidade_quartos": 3,
        "extra_field": "should_be_ignored"
    }
    
    house_features = HouseFeatures(**data)
    
    assert house_features.tamanho == 150.5
    assert house_features.localizacao == "São Paulo"
    assert house_features.quantidade_quartos == 3


def test_house_features_zero_values():
    """Test HouseFeatures with zero values."""
    data = {
        "tamanho": 0.0,
        "localizacao": "Test",
        "quantidade_quartos": 0
    }
    
    house_features = HouseFeatures(**data)
    
    assert house_features.tamanho == 0.0
    assert house_features.localizacao == "Test"
    assert house_features.quantidade_quartos == 0


def test_house_features_negative_values():
    """Test HouseFeatures with negative values."""
    data = {
        "tamanho": -50.0,
        "localizacao": "Test",
        "quantidade_quartos": -1
    }
    
    house_features = HouseFeatures(**data)
    
    assert house_features.tamanho == -50.0
    assert house_features.localizacao == "Test"
    assert house_features.quantidade_quartos == -1


def test_price_prediction_valid_data():
    """Test PricePrediction with valid data returns correct value."""
    data = {"price": 500000.0}
    
    price_prediction = PricePrediction(**data)
    
    assert price_prediction.price == 500000.0


def test_price_prediction_invalid_price():
    """Test PricePrediction raises ValidationError when price is invalid."""
    data = {"price": "invalid"}
    
    with pytest.raises(ValidationError) as excinfo:
        PricePrediction(**data)
    
    assert "price" in str(excinfo.value)


def test_price_prediction_missing_required_field():
    """Test PricePrediction raises ValidationError when required field is missing."""
    data = {}
    
    with pytest.raises(ValidationError) as excinfo:
        PricePrediction(**data)
    
    assert "price" in str(excinfo.value)


def test_price_prediction_negative_price():
    """Test PricePrediction with negative price."""
    data = {"price": -1000.0}
    
    price_prediction = PricePrediction(**data)
    
    assert price_prediction.price == -1000.0


def test_price_prediction_zero_price():
    """Test PricePrediction with zero price."""
    data = {"price": 0.0}
    
    price_prediction = PricePrediction(**data)
    
    assert price_prediction.price == 0.0


def test_price_prediction_large_price():
    """Test PricePrediction with large price value."""
    data = {"price": 999999999.99}
    
    price_prediction = PricePrediction(**data)
    
    assert price_prediction.price == 999999999.99 