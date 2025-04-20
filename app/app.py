import os
from typing import List, Optional
import time
from loguru import logger

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# OpenTelemetry imports for tracing
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import get_tracer_provider, set_tracer_provider

# OpenTelemetry imports for metrics (Prometheus)
from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import start_http_server

# Set up tracing
resource = Resource.create({SERVICE_NAME: "bank-marketing-model-api"})
trace_provider = TracerProvider(resource=resource)
set_tracer_provider(trace_provider)

# Configure Jaeger exporter to connect to your existing Jaeger instance
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",  # Use the host where Jaeger is running
    agent_port=6831,  # This is the default Jaeger agent port
)
span_processor = BatchSpanProcessor(jaeger_exporter)
trace_provider.add_span_processor(span_processor)
tracer = get_tracer_provider().get_tracer("bank-model", "1.0.0")

# Start Prometheus metrics server on port 8099
# If your Prometheus is already configured to scrape from this port
try:
    start_http_server(port=8099, addr="0.0.0.0")
    logger.info("Started Prometheus metrics server on port 8099")
except Exception as e:
    logger.warning(f"Could not start Prometheus metrics server: {e}")

# Set up metrics
reader = PrometheusMetricReader()
meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
set_meter_provider(meter_provider)
meter = metrics.get_meter("bank-model", "1.0.0")

# Create metrics
prediction_counter = meter.create_counter(
    name="prediction_request_counter",
    description="Number of prediction requests"
)

prediction_histogram = meter.create_histogram(
    name="prediction_response_time",
    description="Prediction response time histogram",
    unit="seconds",
)

# Initialize the FastAPI app
app = FastAPI(
    title="Bank Marketing Model API",
    description="API for predicting customer subscription to certificate of deposit",
    version="1.0.0",
)


# Define the input data model based on your features
class CustomerData(BaseModel):
    customer_age: int = Field(
        ..., description="Age of the customer in years", example=41
    )
    tenure: int = Field(
        ...,
        description="How long the customer has been with the bank in months",
        example=24,
    )
    account_balance: float = Field(
        ..., description="Account balance in currency units", example=2567.89
    )
    previous_campaign_number_of_contacts: int = Field(
        0,
        description="Number of contacts performed during previous campaign",
        example=3,
    )
    employment_type: str = Field(
        ..., description="Type of employment", example="private"
    )
    marital_status: str = Field(..., description="Marital status", example="married")
    education_level: str = Field(..., description="Education level", example="tertiary")
    credit_default: str = Field(..., description="Has credit in default?", example="no")
    housing_loan: str = Field(..., description="Has housing loan?", example="yes")
    personal_loan: str = Field(..., description="Has personal loan?", example="no")
    contact_method: str = Field(
        ..., description="Contact communication type", example="cellular"
    )
    previous_campaign_outcome: str = Field(
        "unknown",
        description="Outcome of the previous marketing campaign",
        example="success",
    )


# Define the prediction response model
class PredictionResponse(BaseModel):
    prediction: int = Field(
        ..., description="Prediction (1 = will subscribe, 0 = will not subscribe)"
    )
    probability: float = Field(..., description="Probability of subscription")
    explanation: dict = Field(..., description="Feature importance for this prediction")


# Load the model at startup
@app.on_event("startup")
async def load_model():
    global model, feature_names

    try:
        # Load model from MLflow or direct pickle file
        try:
            # First try loading from MLflow
            # Update this path to your MLflow model location
            # model_uri = "model.pkl"
            # app.model = mlflow.sklearn.load_model(model_uri)

            import joblib

            app.model = joblib.load("./models/model.pkl")
            logger.info("Model loaded successfully")
        except Exception as e:
            # If MLflow loading fails, try direct pickle loading
            # import joblib
            # app.model = joblib.load("model.pkl")
            logger.error(f"Model load unsuccessful: {str(e)}")

        # Try to extract feature names from the model's preprocessing step
        try:
            preprocessor = app.model.named_steps["preprocessor"]
            onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]

            # Get feature names
            cat_feature_names = list(
                onehot.get_feature_names_out(preprocessor.transformers_[1][2])
            )
            numeric_features = preprocessor.transformers_[0][2]
            app.feature_names = list(numeric_features) + cat_feature_names
            logger.info(f"Extracted {len(app.feature_names)} feature names")
        except Exception as e:
            # If feature names can't be extracted, use generic names
            logger.warning(f"Could not extract feature names: {str(e)}")
            app.feature_names = None

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


# Define the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    with tracer.start_as_current_span("predict_single") as span:
        start_time = time.time()
        
        try:
            # Add some attributes to the span
            span.set_attribute("customer.age", customer.customer_age)
            span.set_attribute("customer.education", customer.education_level)
            
            # Convert the input data to a pandas DataFrame
            input_df = pd.DataFrame([customer.dict()])
            
            # Add span event for model prediction
            span.add_event("starting_prediction")
            
            # Make prediction
            prediction = int(app.model.predict(input_df)[0])
            probability = float(app.model.predict_proba(input_df)[0][1])
            
            span.add_event("prediction_complete")
            span.set_attribute("prediction.result", prediction)
            span.set_attribute("prediction.probability", probability)

            # Get feature importances if available
            explanation = {}
            try:
                if (
                    hasattr(app.model, "named_steps")
                    and "classifier" in app.model.named_steps
                ):
                    clf = app.model.named_steps["classifier"]
                    if hasattr(clf, "feature_importances_"):
                        importances = clf.feature_importances_
                        if app.feature_names:
                            for i, importance in enumerate(importances):
                                if i < len(app.feature_names):
                                    explanation[app.feature_names[i]] = float(importance)
                        else:
                            # Use generic feature names
                            for i, importance in enumerate(importances):
                                explanation[f"feature_{i}"] = float(importance)
            except Exception as e:
                span.record_exception(e)
                explanation = {"error": str(e)}

            # Track metrics
            label = {"endpoint": "/predict", "prediction": str(prediction)}
            prediction_counter.add(1, label)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            prediction_histogram.record(elapsed_time, label)
            
            logger.info(f"Prediction completed in {elapsed_time:.4f} seconds")
            
            return {
                "prediction": prediction,
                "probability": probability,
                "explanation": explanation,
            }

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    with tracer.start_as_current_span("health_check"):
        if hasattr(app, "model") and app.model is not None:
            return {"status": "healthy", "model_loaded": True}
        return {"status": "unhealthy", "model_loaded": False}


# Endpoint to get model information
@app.get("/model-info")
async def model_info():
    with tracer.start_as_current_span("model_info"):
        if not hasattr(app, "model"):
            raise HTTPException(status_code=500, detail="Model not loaded")

        info = {"model_type": str(type(app.model)), "features": {}}

        try:
            # Extract preprocessing information
            if (
                hasattr(app.model, "named_steps")
                and "preprocessor" in app.model.named_steps
            ):
                preprocessor = app.model.named_steps["preprocessor"]

                # Get numeric features
                if len(preprocessor.transformers_) > 0:
                    num_transformer = preprocessor.transformers_[0]
                    if num_transformer[0] == "num":
                        info["features"]["numeric"] = list(num_transformer[2])

                # Get categorical features
                if len(preprocessor.transformers_) > 1:
                    cat_transformer = preprocessor.transformers_[1]
                    if cat_transformer[0] == "cat":
                        info["features"]["categorical"] = list(cat_transformer[2])

            # Get classifier information
            if hasattr(app.model, "named_steps") and "classifier" in app.model.named_steps:
                clf = app.model.named_steps["classifier"]
                info["classifier"] = {
                    "type": str(type(clf)),
                    "parameters": clf.get_params(),
                }

        except Exception as e:
            info["error"] = str(e)

        return info


# Batch prediction endpoint
@app.post("/batch-predict")
async def batch_predict(customers: List[CustomerData]):
    with tracer.start_as_current_span("batch_predict") as span:
        start_time = time.time()
        
        if not hasattr(app, "model"):
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise HTTPException(status_code=500, detail="Model not loaded")

        try:
            # Add batch size to span
            span.set_attribute("batch.size", len(customers))
            
            # Convert the input data to a pandas DataFrame
            input_df = pd.DataFrame([customer.dict() for customer in customers])
            
            span.add_event("starting_batch_prediction")
            
            # Make predictions
            predictions = [int(pred) for pred in app.model.predict(input_df)]
            probabilities = [float(prob[1]) for prob in app.model.predict_proba(input_df)]
            
            span.add_event("batch_prediction_complete")
            
            # Track metrics
            label = {"endpoint": "/batch-predict", "batch_size": len(customers)}
            prediction_counter.add(len(customers), label)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            prediction_histogram.record(elapsed_time, label)
            
            logger.info(f"Batch prediction ({len(customers)} items) completed in {elapsed_time:.4f} seconds")
            
            return {"predictions": predictions, "probabilities": probabilities}

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            logger.error(f"Batch prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# Instrument FastAPI app with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)


if __name__ == "__main__":
    # Run the app with uvicorn when this script is executed directly
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)