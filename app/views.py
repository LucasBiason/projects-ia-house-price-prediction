"""
FastAPI Router Views for House Price Prediction API.

This module contains the FastAPI router and endpoint definitions for the house
price prediction service. It provides the main prediction endpoint that accepts
house features and returns predicted prices using the trained machine learning
model.

The views handle request validation, model prediction, and error responses
with proper HTTP status codes and error messages.
"""

from fastapi import APIRouter, HTTPException

from .model import HousePricePrediction
from .schemas import HouseFeatures, PricePrediction

router = APIRouter()

# Global model instance
model = HousePricePrediction()


@router.post("/predict", response_model=PricePrediction)
def predict(features: HouseFeatures) -> PricePrediction:
    """
    Predict house price based on input features.

    This endpoint accepts house features (size, location, number of bedrooms)
    and returns a predicted house price using the trained machine learning
    model. The prediction is based on patterns learned from historical
    house price data.

    Args:
        features (HouseFeatures): House features for prediction including:
            - tamanho: House size in square meters
            - localizacao: House location/neighborhood
            - quantidade_quartos: Number of bedrooms

    Returns:
        PricePrediction: Predicted house price in currency units.

    Raises:
        HTTPException: If the model file is not found or prediction fails.
            - status_code: 500 (Internal Server Error)
            - detail: Error message describing the issue

    Example:
        >>> response = await predict(HouseFeatures(
        ...     tamanho=400.0,
        ...     localizacao="Centro",
        ...     quantidade_quartos=2
        ... ))
        >>> print(f"Predicted price: R$ {response.price:,.2f}")
    """
    try:
        prediction: float = model.predict([
            features.tamanho,
            features.localizacao,
            features.quantidade_quartos
        ])
        return PricePrediction(price=prediction)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))