"""
House Price Prediction Service Main Application.

This module contains the main FastAPI application for house price prediction
service with proper initialization and error handling. It provides endpoints
for health checks, service status, and integrates the machine learning model
for house price predictions.

The application uses async context managers for lifecycle management and
includes proper error handling and logging throughout the service.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .model import HousePricePrediction
from .views import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: HousePricePrediction | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown operations.
    
    This context manager handles the initialization and cleanup of the
    application, including model loading and training during startup,
    and proper shutdown procedures.
    
    Args:
        app: The FastAPI application instance.
        
    Yields:
        None: Control is yielded to the application runtime.
        
    Raises:
        Exception: If model initialization fails during startup.
        
    Example:
        async with lifespan(app):
            # Application runs here
            pass
    """
    global model
    try:
        logger.info("Initializing HousePricePrediction model...")
        model = HousePricePrediction()
        model.train()
        logger.info("HousePricePrediction model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize HousePricePrediction model: {e}")
        raise
    
    yield
    
    logger.info("Shutting down House Price Prediction service...")


app = FastAPI(
    title="House Price Prediction Service",
    description="AI-powered house price prediction service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint to verify service status and availability.
    
    This endpoint provides basic information about the service status,
    including whether the machine learning model is ready for predictions.
    
    Returns:
        Dict[str, Any]: Service status information containing:
            - message: Service availability message
            - status: Current service status
            - model_ready: Boolean indicating if model is initialized
            
    Example:
        >>> response = await root()
        >>> print(response["status"])
        "healthy"
    """
    return {
        "message": "House Price Prediction Service is online!",
        "status": "healthy",
        "model_ready": model is not None
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for service monitoring.
    
    This endpoint performs a comprehensive health check of the service,
    verifying that the machine learning model is properly initialized
    and ready to handle prediction requests.
    
    Returns:
        Dict[str, Any]: Health status information containing:
            - status: Service health status
            - model_ready: Boolean indicating if model is ready
            
    Raises:
        HTTPException: If the model is not ready or service is unhealthy.
            - status_code: 503 (Service Unavailable)
            - detail: Error message describing the issue
            
    Example:
        >>> response = await health_check()
        >>> print(response["status"])
        "healthy"
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not ready")
        
        return {
            "status": "healthy",
            "model_ready": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")