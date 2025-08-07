 # Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

### Added
- **Core Application Structure**
  - FastAPI application with proper lifecycle management
  - Machine learning model for house price prediction using Linear Regression
  - RESTful API endpoints for health checks and predictions
  - Comprehensive data preprocessing pipeline with StandardScaler and OneHotEncoder

- **API Endpoints**
  - `GET /` - Root endpoint for service status and availability
  - `GET /health` - Health check endpoint for service monitoring
  - `POST /predict` - House price prediction endpoint

- **Machine Learning Model**
  - `HousePricePrediction` class with training and prediction capabilities
  - Support for numerical features (size, bedrooms) and categorical features (location)
  - Automatic model persistence and loading
  - Data preprocessing pipeline with scikit-learn

- **Data Validation**
  - Pydantic schemas for request/response validation
  - `HouseFeatures` model for input validation
  - `PricePrediction` model for output validation
  - Automatic API documentation generation

- **Testing Infrastructure**
  - Comprehensive unit test suite with 100% code coverage
  - Test structure mirroring application structure (`tests/app/`)
  - Pytest configuration with coverage reporting
  - Mock-based testing for external dependencies

- **Development Tools**
  - Docker containerization for consistent development environment
  - Docker Compose configuration for testing and development
  - Makefile with common development commands
  - Requirements management with pinned dependencies

- **Code Quality**
  - Complete docstrings following Google/NumPy style
  - Type hints for all functions, methods, and variables
  - Organized imports (standard → third-party → local)
  - PEP 8 compliance and code formatting

### Changed
- **Dependency Management**
  - Updated all dependencies to latest compatible versions
  - Resolved Python version compatibility issues (3.11+)
  - Fixed dependency conflicts (pydantic-core, fastapi, coverage)

- **Docker Configuration**
  - Updated base image from Python 3.9 to Python 3.11
  - Fixed Dockerfile paths and permissions
  - Resolved port conflicts in docker-compose.yml

- **Testing Framework**
  - Restructured test organization to mirror app structure
  - Enhanced test coverage with edge cases and error scenarios
  - Improved test reliability and reproducibility

- **Code Documentation**
  - Added comprehensive module docstrings
  - Implemented class and method documentation
  - Translated Portuguese comments to English
  - Added practical examples in docstrings

### Fixed
- **Dependency Issues**
  - Resolved numpy/pandas compatibility with Python 3.13
  - Fixed pydantic-core version conflicts
  - Corrected httpx and pytest-asyncio dependencies

- **Docker Build Issues**
  - Fixed entrypoint.sh path issues
  - Resolved Python version mismatches
  - Corrected chmod permissions

- **Test Failures**
  - Fixed test assertions for new response format
  - Corrected mock configurations
  - Resolved async test issues

- **Code Quality Issues**
  - Fixed line length violations
  - Corrected import organization
  - Resolved type hint issues
