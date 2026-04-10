# SkyGuard: Flight Disruption Prediction System

A production-quality machine learning system designed to predict flight cancellations and disruptions using CatBoost models and secondary weather data integration.

## 🚀 Overview

SkyGuard is a full-stack ML solution developed to assist travelers and airline operators in identifying high-risk flight schedules. By merging historical flight performance data with real-time weather indicators (origin/destination temperature, visibility, wind speed), the system provides probabilistic forecasts of flight cancellations and delay risks.

## ✨ Key Features

- **Predictive Analytics**: Real-time inference using a high-performance CatBoost classification model.
- **Automated Data Pipeline**: Robust ETL module for cleaning weather station data and merging it with flight logs.
- **Production-Ready API**: FastAPI-based microservice with structured logging, health monitoring, and performance tracking.
- **Containerized Deployment**: Ready-to-use Docker environment for seamless scaling.
- **Comprehensive Logging**: Centralized logging system for auditing inference requests and internal processing.

## 🛠 Tech Stack

- **Language**: Python 3.10+
- **Machine Learning**: CatBoost, Scikit-Learn, Joblib
- **Data Engineering**: Pandas, NumPy
- **API Framework**: FastAPI, Uvicorn, Pydantic
- **DevOps**: Docker, Git

## 📂 Project Structure

```text
Delayed-Flight/
├── data/                   # Dataset storage (Raw, Processed, Mapping)
├── models/                 # Pre-trained ML model binaries (.cbm, .joblib)
├── notebooks/              # Exploratory Data Analysis & Research
├── src/
│   ├── api/                # FastAPI application & Pydantic schemas
│   ├── core/               # ML inference logic & Data pipelines
│   ├── config/             # Centralized project settings
│   └── utils/              # Logging & shared helpers
├── tests/                  # API and unit tests
├── Dockerfile              # Container definition
├── requirements.txt        # Dependency management
└── README.md               # Project documentation
```

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Delayed-Flight
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the API Locally
```bash
uvicorn src.api.app:app --reload
```
The API will be available at `http://localhost:8000`. You can access the interactive documentation at `http://localhost:8000/docs`.

### 4. Run via Docker
```bash
docker build -t skyguard-api .
docker run -p 8000:8000 skyguard-api
```

## 🚦 Example Usage

### Predict Flight Cancellation
**POST** `/predict`

**Request Body:**
```json
{
  "Origin": "ATL",
  "Dest": "TPA",
  "CRSDepTime": 1950,
  "CRSArrTime": 2118,
  "Distance": 406.0,
  "Month": 12,
  "DayOfWeek": 1,
  "IATA_Code_Marketing_Airline": "DL"
}
```

**Response:**
```json
{
  "will_cancel": false,
  "cancel_probability": 0.042,
  "threshold": 0.5,
  "input_received": { ... },
  "metadata": {
    "model_version": "catboost-v1",
    "features_used": 14
  }
}
```

## 📝 Author
Genuinely built and refactored by a final-year student with a passion for Data Science and Software Architecture.
