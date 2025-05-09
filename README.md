# Bank Marketing Campaign API

This repository contains a FastAPI application that serves a machine learning model for predicting whether a customer will subscribe to a certificate of deposit (CD).

## Project Structure

```
.
├── data/                     # Data directory
│   ├── bank-names.txt        # List of bank names
│   └── bank.xlsx             # Bank data in Excel format
├── models/                   # Model artifacts directory
│   └── model.pkl             # Serialized machine learning model
├── notebooks/                # Jupyter notebooks for analysis and exploration
├── tests/                    # Test files and test cases
    └── test_model_correctness.py                
├── training/                 # Model training code
│   └── bank-ml-model-mlflow.py  # MLflow-based training script for bank model
├── .pre-commit-config.yaml   # Pre-commit config
├── app.py                    # Main application entry point
├── bank-campaign-docker-compose.yaml  # Docker Compose file 
├── Dockerfile                # Docker configuration for containerization
├── Jenkinsfile               # Jenkins CI/CD pipeline configuration
├── Readme.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Setup Instructions

### Option 1: Running locally

1. **Prerequisites**:
   - Python 3.9+
   - pip

2. **Installation**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd bank-campaign-api
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Running the API**:
   ```bash
   # Make sure model.pkl file is in the current directory
   uvicorn app:app --reload
   ```

   The API will be available at `http://localhost:8000`.

### Option 2: Using Docker

1. **Prerequisites**:
   - Docker

2. **Building and running the Docker container**:
   ```bash
   # Build the Docker image
   docker build -t ducngminh/bank-campaign-api:<version> .
   
   # Run the container
   docker run -p 8000:8000 ducngminh/bank-campaign-api:<version>
   ```

   The API will be available at `http://localhost:8000`.

## API Endpoints

### 1. Make a prediction for a single customer

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
  "customer_age": 41,
  "tenure": 24,
  "account_balance": 2567.89,
  "previous_campaign_number_of_contacts": 3,
  "employment_type": "private",
  "marital_status": "married",
  "education_level": "tertiary",
  "credit_default": "no",
  "housing_loan": "yes",
  "personal_loan": "no",
  "contact_method": "cellular",
  "previous_campaign_outcome": "success"
}
```

**Response**:
```json
{
  "prediction": 1,
  "probability": 0.85,
  "explanation": {
    "feature1": 0.3,
    "feature2": 0.2,
    "...": "..."
  }
}
```

### 2. Make predictions for multiple customers

**Endpoint**: `POST /batch-predict`

**Request Body**:
```json
[
  {
    "customer_age": 41,
    "tenure": 24,
    "...": "..."
  },
  {
    "customer_age": 35,
    "tenure": 12,
    "...": "..."
  }
]
```

**Response**:
```json
{
  "predictions": [1, 0],
  "probabilities": [0.85, 0.23]
}
```

### 3. Get model information

**Endpoint**: `GET /model-info`

**Response**:
```json
{
  "model_type": "sklearn.pipeline.Pipeline",
  "features": {
    "numeric": ["customer_age", "tenure", "account_balance", "previous_campaign_number_of_contacts"],
    "categorical": ["employment_type", "marital_status", "education_level", "credit_default", "housing_loan", "personal_loan", "contact_method", "previous_campaign_outcome"]
  },
  "classifier": {
    "type": "sklearn.ensemble._forest.RandomForestClassifier",
    "parameters": {
      "n_estimators": 100,
      "random_state": 42,
      "class_weight": "balanced"
    }
  }
}
```

### 4. Health check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Testing the API

You can test the API using the provided `test_client.py` script:

```bash
python test_client.py
```

Or you can use curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"customer_age": 41, "tenure": 24, "account_balance": 2567.89, "previous_campaign_number_of_contacts": 3, "employment_type": "private", "marital_status": "married", "education_level": "tertiary", "credit_default": "no", "housing_loan": "yes", "personal_loan": "no", "contact_method": "cellular", "previous_campaign_outcome": "success"}'
```

## Interactive API Documentation

FastAPI automatically generates interactive API documentation.

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment Considerations

For production deployment, consider:

1. Setting up proper authentication
2. Configuring HTTPS
3. Implementing rate limiting
4. Setting up monitoring and logging
5. Using a production-grade ASGI server like Gunicorn with Uvicorn workers
