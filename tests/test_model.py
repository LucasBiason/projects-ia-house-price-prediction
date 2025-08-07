"""
Unit tests for app.model module.
"""
import os
import pickle
import pandas as pd
import pytest
from unittest.mock import patch, Mock, mock_open

from app.model import HousePricePrediction


def test_house_price_prediction_init():
    """Test HousePricePrediction initialization sets correct default values."""
    model = HousePricePrediction()
    
    assert model.model_path == 'model.pkl'
    assert model.pipeline is None


def test_load_data_success():
    """Test load_data method successfully loads CSV data."""
    mock_data = pd.DataFrame({
        'tamanho': [100, 150, 200],
        'localizacao': ['SP', 'RJ', 'MG'],
        'quantidade_quartos': [2, 3, 4],
        'price': [300000, 450000, 600000]
    })
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = mock_data
        
        model = HousePricePrediction()
        result = model.load_data()
        
        assert result.equals(mock_data)
        mock_read_csv.assert_called_once_with('data/house_prices.csv', sep='|')


def test_load_data_file_not_found():
    """Test load_data method raises FileNotFoundError when file not found."""
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        model = HousePricePrediction()
        
        with pytest.raises(FileNotFoundError) as excinfo:
            model.load_data()
        
        assert "File not found" in str(excinfo.value)


def test_train_success():
    """Test train method successfully trains the model."""
    mock_data = pd.DataFrame({
        'tamanho': [100, 150, 200],
        'localizacao': ['SP', 'RJ', 'MG'],
        'quantidade_quartos': [2, 3, 4],
        'price': [300000, 450000, 600000]
    })
    
    with patch('pandas.read_csv') as mock_read_csv, \
         patch('sklearn.pipeline.Pipeline.fit') as mock_fit, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.dump') as mock_pickle_dump:
        
        mock_read_csv.return_value = mock_data
        
        model = HousePricePrediction()
        model.train()
        
        mock_read_csv.assert_called_once()
        mock_fit.assert_called_once()
        mock_file.assert_called_once_with('model.pkl', 'wb')
        mock_pickle_dump.assert_called_once()


def test_train_data_loading_error():
    """Test train method raises FileNotFoundError when data loading fails."""
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.side_effect = FileNotFoundError("Data file not found")
        
        model = HousePricePrediction()
        
        with pytest.raises(FileNotFoundError) as excinfo:
            model.train()
        
        assert "Data file not found" in str(excinfo.value)


def test_predict_success_with_list():
    """Test predict method with list input returns correct prediction."""
    mock_pipeline = Mock()
    mock_pipeline.predict.return_value = [500000.0]
    
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.load') as mock_pickle_load:
        
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_pipeline
        
        model = HousePricePrediction()
        features = [150.5, 'São Paulo', 3]
        
        result = model.predict(features)
        
        assert result == 500000.0
        mock_exists.assert_called_once_with('model.pkl')
        mock_file.assert_called_once_with('model.pkl', 'rb')
        mock_pickle_load.assert_called_once()


def test_predict_success_with_dataframe():
    """Test predict method with DataFrame input returns correct prediction."""
    mock_pipeline = Mock()
    mock_pipeline.predict.return_value = [750000.0]
    
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.load') as mock_pickle_load:
        
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_pipeline
        
        model = HousePricePrediction()
        features = pd.DataFrame({
            'tamanho': [200.0],
            'localizacao': ['Rio de Janeiro'],
            'quantidade_quartos': [4]
        })
        
        result = model.predict(features)
        
        assert result == 750000.0
        mock_exists.assert_called_once_with('model.pkl')
        mock_file.assert_called_once_with('model.pkl', 'rb')


def test_predict_file_not_found():
    """Test predict method raises FileNotFoundError when model file not found."""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False
        
        model = HousePricePrediction()
        features = [150.5, 'São Paulo', 3]
        
        with pytest.raises(FileNotFoundError) as excinfo:
            model.predict(features)
        
        assert "Model file 'model.pkl' not found" in str(excinfo.value)
        mock_exists.assert_called_once_with('model.pkl')


def test_predict_with_different_input_types():
    """Test predict method with different input types."""
    mock_pipeline = Mock()
    mock_pipeline.predict.return_value = [300000.0]
    
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.load') as mock_pickle_load:
        
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_pipeline
        
        model = HousePricePrediction()
        
        test_cases = [
            ([100.0, 'SP', 2], 300000.0),
            ([0.0, 'Test', 0], 300000.0),
            ([-50.0, 'Test', -1], 300000.0),
        ]
        
        for features, expected in test_cases:
            result = model.predict(features)
            assert result == expected


def test_predict_pipeline_load_error():
    """Test predict method raises PickleError when pipeline loading fails."""
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.load') as mock_pickle_load:
        
        mock_exists.return_value = True
        mock_pickle_load.side_effect = pickle.PickleError("Invalid pickle file")
        
        model = HousePricePrediction()
        features = [150.5, 'São Paulo', 3]
        
        with pytest.raises(pickle.PickleError) as excinfo:
            model.predict(features)
        
        assert "Invalid pickle file" in str(excinfo.value)


def test_predict_pipeline_prediction_error():
    """Test predict method raises Exception when pipeline prediction fails."""
    mock_pipeline = Mock()
    mock_pipeline.predict.side_effect = Exception("Prediction error")
    
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.load') as mock_pickle_load:
        
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_pipeline
        
        model = HousePricePrediction()
        features = [150.5, 'São Paulo', 3]
        
        with pytest.raises(Exception) as excinfo:
            model.predict(features)
        
        assert "Prediction error" in str(excinfo.value)


def test_predict_with_large_values():
    """Test predict method with large feature values."""
    mock_pipeline = Mock()
    mock_pipeline.predict.return_value = [999999999.99]
    
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.load') as mock_pickle_load:
        
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_pipeline
        
        model = HousePricePrediction()
        features = [999999.99, 'Large City', 999]
        
        result = model.predict(features)
        
        assert result == 999999999.99


def test_predict_with_decimal_values():
    """Test predict method with decimal feature values."""
    mock_pipeline = Mock()
    mock_pipeline.predict.return_value = [500123.45]
    
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()) as mock_file, \
         patch('pickle.load') as mock_pickle_load:
        
        mock_exists.return_value = True
        mock_pickle_load.return_value = mock_pipeline
        
        model = HousePricePrediction()
        features = [150.75, 'São Paulo', 3]
        
        result = model.predict(features)
        
        assert result == 500123.45 