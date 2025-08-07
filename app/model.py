"""
House Price Prediction Model Module.

This module contains the HousePricePrediction class that handles the machine
learning model for predicting house prices. It includes data loading, model
training, and prediction functionality using scikit-learn's LinearRegression
with preprocessing pipeline for handling both numerical and categorical features.

The model uses a combination of StandardScaler for numerical features and
OneHotEncoder for categorical features, wrapped in a scikit-learn Pipeline
for consistent preprocessing and prediction.
"""

import os
import pickle
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class HousePricePrediction:
    """
    House Price Prediction Model using Linear Regression.

    This class provides functionality to train and use a machine learning model
    for predicting house prices based on features like size, location, and
    number of bedrooms. The model includes preprocessing steps for both
    numerical and categorical features.

    Attributes:
        model_path (str): Path to the saved model file.
        pipeline (Pipeline | None): The trained scikit-learn pipeline containing
            preprocessing and model components.

    Example:
        >>> model = HousePricePrediction()
        >>> model.train()
        >>> prediction = model.predict([400, 'Centro', 2])
        >>> print(f"Predicted price: R$ {prediction:,.2f}")
    """

    def __init__(self) -> None:
        """
        Initialize the HousePricePrediction model.

        Sets up the model path and initializes the pipeline as None.
        The model will be trained and saved to 'model.pkl' file.
        """
        self.model_path: str = 'model.pkl'
        self.pipeline: Pipeline | None = None

    def load_data(self) -> pd.DataFrame:
        """
        Load house price data from CSV file.

        Reads the house prices dataset from the data directory. The CSV file
        should contain columns for house features and prices.

        Returns:
            pd.DataFrame: Loaded house price data with columns including
                'tamanho', 'localizacao', 'quantidade_quartos', and 'price'.

        Raises:
            FileNotFoundError: If the data file 'data/house_prices.csv' is not found.
            pd.errors.EmptyDataError: If the CSV file is empty.
            pd.errors.ParserError: If the CSV file has invalid format.

        Example:
            >>> data = model.load_data()
            >>> print(f"Loaded {len(data)} records")
        """
        data: pd.DataFrame = pd.read_csv('data/house_prices.csv', sep='|')
        return data

    def train(self) -> None:
        """
        Train the house price prediction model.

        This method loads the training data, creates a preprocessing pipeline
        with StandardScaler for numerical features and OneHotEncoder for categorical
        features, trains a LinearRegression model, and saves the trained pipeline
        to disk.

        The preprocessing pipeline handles:
        - Numerical features (tamanho, quantidade_quartos): StandardScaler normalization
        - Categorical features (localizacao): OneHotEncoder for encoding

        The data is split into 80% training and 20% testing sets for model validation.

        Raises:
            FileNotFoundError: If the data file cannot be loaded.
            ValueError: If the data contains invalid values or missing required columns.
            Exception: If model training fails due to insufficient data or other issues.

        Example:
            >>> model = HousePricePrediction()
            >>> model.train()
            >>> print("Model trained and saved successfully")
        """
        data: pd.DataFrame = self.load_data()
        X: pd.DataFrame = data.drop('price', axis=1)
        y: pd.Series = data['price']
        
        # Create preprocessor for normalization and encoding
        preprocessor: ColumnTransformer = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['tamanho', 'quantidade_quartos']),  # Numerical features
                ('cat', OneHotEncoder(), ['localizacao'])  # Categorical features
            ])
        
        # Create pipeline with preprocessor and model
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        
        # Split data into training and testing sets
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Save the trained pipeline
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def predict(self, features: Union[List[Union[float, str, int]], pd.DataFrame]) -> float:
        """
        Predict house price based on input features.

        Loads the trained model from disk and makes a prediction for the given
        house features. The features should include size, location, and number
        of bedrooms.

        Args:
            features: House features for prediction. Can be either:
                - List containing [tamanho, localizacao, quantidade_quartos]
                - pandas DataFrame with columns ['tamanho', 'localizacao', 'quantidade_quartos']

        Returns:
            float: Predicted house price rounded to 2 decimal places.

        Raises:
            FileNotFoundError: If the trained model file is not found.
            ValueError: If features are in incorrect format or missing required columns.
            Exception: If prediction fails due to model loading or inference issues.

        Example:
            >>> model = HousePricePrediction()
            >>> price = model.predict([400, 'Centro', 2])
            >>> print(f"Predicted price: R$ {price:,.2f}")
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file '{self.model_path}' not found. Please train the model first."
            )
        
        with open(self.model_path, 'rb') as f:
            self.pipeline = pickle.load(f)
        
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(
                [features], 
                columns=['tamanho', 'localizacao', 'quantidade_quartos']
            )
        
        prediction: np.ndarray = self.pipeline.predict(features)
        return round(float(prediction[0]), 2)