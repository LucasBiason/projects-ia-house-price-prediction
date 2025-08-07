"""
Pydantic Schema Models for House Price Prediction API.

This module defines the Pydantic models used for request and response validation
in the house price prediction API. It includes schemas for house features input
and price prediction output to ensure data consistency and type safety.

The schemas provide automatic validation, serialization, and documentation
generation for the FastAPI endpoints.
"""

from pydantic import BaseModel


class HouseFeatures(BaseModel):
    """
    House features schema for prediction requests.

    This model defines the structure and validation rules for house features
    that are used to predict house prices. All fields are required and have
    specific type constraints.

    Attributes:
        tamanho (float): House size in square meters. Must be a positive number.
        localizacao (str): House location/neighborhood. Must be a non-empty string.
        quantidade_quartos (int): Number of bedrooms. Must be a positive integer.

    Example:
        >>> features = HouseFeatures(
        ...     tamanho=400.0,
        ...     localizacao="Centro",
        ...     quantidade_quartos=2
        ... )
        >>> print(features.tamanho)
        400.0
    """

    tamanho: float
    localizacao: str
    quantidade_quartos: int


class PricePrediction(BaseModel):
    """
    Price prediction response schema.

    This model defines the structure for the API response containing the
    predicted house price. The price is returned as a float value.

    Attributes:
        price (float): Predicted house price in currency units.

    Example:
        >>> prediction = PricePrediction(price=896983.92)
        >>> print(f"Predicted price: R$ {prediction.price:,.2f}")
        Predicted price: R$ 896,983.92
    """

    price: float