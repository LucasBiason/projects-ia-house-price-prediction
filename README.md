# House Price Prediction Service

A machine learning service for predicting house prices based on features like size, location, and number of bedrooms.

## 🚀 Features

- **Machine Learning Model**: Linear Regression with preprocessing pipeline
- **RESTful API**: FastAPI-based service with automatic documentation
- **Data Validation**: Pydantic models for request/response validation
- **Comprehensive Testing**: 100% code coverage with comprehensive test suite
- **Docker Support**: Containerized application for consistent deployment
- **Code Quality**: Type hints, docstrings, and organized imports

## 📋 Requirements

- Python 3.11+
- Docker and Docker Compose

## 🛠️ Installation

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd projects-ia-house-price-prediction
   ```

2. **Run the application**
   ```bash
   make runapp
   ```

3. **Access the API**
   - API Documentation: http://localhost:9000/docs
   - Health Check: http://localhost:9000/health
   - Root Endpoint: http://localhost:9000/

### Local Development

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   uvicorn app.main:app --reload
   ```

## 🧪 Testing

### Run all tests
```bash
make test
```

### Run specific test modules
```bash
# Test main application
docker compose run --rm test pytest tests/test_main.py -v

# Test machine learning model
docker compose run --rm test pytest tests/test_model.py -v

# Test API endpoints
docker compose run --rm test pytest tests/test_views.py -v

# Test data validation
docker compose run --rm test pytest tests/test_schemas.py -v
```

### Coverage Report
```bash
docker compose run --rm test pytest tests/ --cov=app --cov-report=term-missing
```

## 📚 API Documentation

### Endpoints

#### `GET /`
Returns service status and availability information.

**Response:**
```json
{
  "message": "House Price Prediction Service is online!",
  "status": "healthy",
  "model_ready": true
}
```

#### `GET /health`
Performs comprehensive health check of the service.

**Response:**
```json
{
  "status": "healthy",
  "model_ready": true
}
```

#### `POST /predict`
Predicts house price based on input features.

**Request:**
```json
{
  "tamanho": 400.0,
  "localizacao": "Centro",
  "quantidade_quartos": 2
}
```

**Response:**
```json
{
  "price": 896983.92
}
```

## 🔧 Development

### Project Structure
```
projects-ia-house-price-prediction/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   ├── model.py         # Machine learning model
│   ├── schemas.py       # Pydantic models
│   └── views.py         # API endpoints
├── data/
│   └── house_prices.csv # Training data
├── tests/
│   ├── test_main.py     # Application tests
│   ├── test_model.py    # Model tests
│   ├── test_schemas.py  # Schema tests
│   └── test_views.py    # Endpoint tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

### Available Commands

```bash
# Run application
make runapp

# Run application in development mode
make runapp-dev

# Run tests
make test

# Code formatting and linting
make lint
```

### Machine Learning Model

The service uses a Linear Regression model with the following preprocessing:

- **Numerical Features**: StandardScaler normalization
- **Categorical Features**: OneHotEncoder encoding
- **Pipeline**: scikit-learn Pipeline for consistent preprocessing

### Data Format

Training data should be in CSV format with the following columns:
- `tamanho`: House size in square meters
- `localizacao`: House location/neighborhood
- `quantidade_quartos`: Number of bedrooms
- `price`: Target variable (house price)

## 🐛 Troubleshooting

### Common Issues

1. **Docker build fails**
   - Ensure Docker is running
   - Check available disk space

2. **Tests fail**
   - Run `docker compose down` to clean containers
   - Rebuild with `docker compose build --no-cache`

3. **Model not found error**
   - Ensure training data exists in `data/house_prices.csv`
   - Run model training first

4. **Port conflicts**
   - Change port in `docker-compose.yml` if 9000 is in use

### Logs

View application logs:
```bash
docker compose logs web
```

View test logs:
```bash
docker compose logs test
```


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure 100% test coverage
6. Submit a pull request

### Code Standards

- Follow PEP 8 style guide
- Use type hints for all functions
- Add docstrings for all modules, classes, and functions
- Organize imports (standard → third-party → local)
- Write comprehensive tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆕 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.
