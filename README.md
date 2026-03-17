# Tabular ML — Credit Card Fraud Detection

Production-grade ML pipeline for credit card fraud detection with full MLOps tooling. Demonstrates end-to-end workflow from exploratory data analysis to deployed inference service with monitoring.

## Overview

Binary classification on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 0.172% fraud rate). The project implements:

- **Exploratory Data Analysis** with distribution analysis, correlation heatmaps, and fraud pattern investigation
- **Feature Engineering** pipeline with cyclical time encoding, log-transformed amounts, and interaction features
- **Model Training & Tuning** of XGBoost, LightGBM, and CatBoost with Optuna hyperparameter optimization
- **Ensemble Methods** including stacking and blending for improved performance
- **MLOps Infrastructure** with MLflow experiment tracking, FastAPI inference service, and Docker containerization
- **Production Monitoring** with Evidently for data drift detection and simulation

## Results

| Model | PR-AUC (Test) | ROC-AUC (Test) | F1 Score | Precision | Recall | Optimal Threshold |
|-------|---------------|----------------|----------|-----------|--------|-------------------|
| XGBoost | 0.867 | 0.977 | 0.882 | 0.932 | 0.837 | 0.817 |
| LightGBM | 0.864 | 0.971 | 0.877 | 0.921 | 0.837 | 0.732 |
| CatBoost | 0.838 | 0.976 | 0.827 | 0.827 | 0.827 | 0.949 |
| Stacking Ensemble | 0.862 | 0.978 | 0.878 | 0.912 | 0.847 | 0.932 |
| Blending Ensemble | 0.867 | 0.977 | 0.882 | 0.932 | 0.837 | 0.817 |

**Key Findings:**
- XGBoost achieved the best PR-AUC (0.867) on test data, critical for imbalanced fraud detection
- Stacking ensemble achieved highest recall (0.847), balancing precision and recall
- All models show strong ROC-AUC (>0.97), indicating excellent discrimination ability
- Class imbalance handled via `scale_pos_weight` and optimized thresholds

## Stack

| Category | Tools |
|---|---|
| **Models** | XGBoost, LightGBM, CatBoost |
| **Hyperparameter Tuning** | Optuna |
| **Experiment Tracking** | MLflow |
| **API & Serving** | FastAPI, Uvicorn |
| **Monitoring** | Evidently |
| **Containerization** | Docker, Docker Compose |
| **Core Data Science** | pandas, scikit-learn, numpy |
| **Testing** | pytest, httpx |

## Project Structure

```
tabular-ml/
├── src/tabular_ml/              # Main Python package
│   ├── data/                    # Data loading & splitting
│   │   └── loader.py
│   ├── features/                # Feature engineering
│   │   ├── engineering.py       # Custom sklearn transformers
│   │   └── pipeline.py          # Pipeline orchestration
│   ├── models/                  # Training & evaluation
│   │   ├── trainer.py           # Model training with MLflow
│   │   ├── tuning.py            # Optuna hyperparameter tuning
│   │   ├── ensemble.py          # Stacking/blending ensembles
│   │   ├── evaluation.py        # Metrics & threshold optimization
│   │   └── train_all.py         # Full training pipeline
│   ├── api/                     # FastAPI inference service
│   │   ├── app.py               # FastAPI application
│   │   └── schemas.py           # Pydantic request/response models
│   └── monitoring/              # Data drift monitoring
│       ├── drift.py             # Evidently drift detection
│       └── __init__.py
├── notebooks/                   # Exploratory analysis
│   └── 01_eda.ipynb            # EDA notebook with 6+ visualizations
├── configs/                     # Configuration files
│   └── default.yaml            # Pipeline configuration
├── artifacts/                   # Serialized models & results
│   ├── preprocessing_pipeline.joblib
│   ├── xgboost_model.joblib    # Best individual model
│   ├── training_results.json   # Individual model metrics
│   ├── ensemble_results.json   # Ensemble metrics
│   └── drift_detection_report.html
├── data/                        # Dataset (gitignored)
│   ├── raw/                    # Original CSV from Kaggle
│   └── processed/              # Engineered features
├── scripts/                     # Utility scripts
│   └── monitoring_demo.py      # Drift monitoring demonstration
├── tests/                       # Comprehensive test suite (69 tests)
├── tasks/                       # Project management
│   ├── todo.md                 # Phase tracking
│   └── lessons.md              # Development lessons
├── Dockerfile                  # Multi-stage production container
├── docker-compose.yml          # API + MLflow services
├── requirements.txt            # Development dependencies
├── requirements-docker.txt     # Minimal runtime dependencies
└── pyproject.toml              # Python package configuration
```

## Quick Start

### 1. Clone & Setup

```bash
# Clone repository
git clone <repo-url>
cd tabular-ml

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (requires Kaggle API key)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw/ --unzip
```

### 2. Run Exploratory Data Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_eda.ipynb
```

The EDA notebook includes:
- Dataset statistics and missing value analysis
- Class distribution visualization (0.172% fraud rate)
- Feature distributions and correlation analysis
- Fraud vs legitimate transaction comparisons
- Modeling implications summary

### 3. Train Models

```bash
# Run full training pipeline
python -m tabular_ml.models.train_all

# Or run individual components
python -c "
from tabular_ml.features.pipeline import fit_and_transform, save_pipeline
from tabular_ml.models.trainer import train_xgboost

# Feature engineering
splits, pipeline = fit_and_transform()
save_pipeline(pipeline)

# Model training
model, metrics = train_xgboost(splits['train'], splits['val'], splits['test'])
print(f'XGBoost PR-AUC: {metrics[\"test_pr_auc\"]:.3f}')
"
```

### 4. Start Inference API

```bash
# Start FastAPI server
uvicorn tabular_ml.api.app:app --reload --port 8000

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'
```

### 5. Monitor Data Drift

```bash
# Run monitoring demonstration
python scripts/monitoring_demo.py

# Or use directly
python -c "
from tabular_ml.monitoring.drift import detect_data_drift, simulate_drift
import pandas as pd

# Load your data
reference = pd.read_csv('data/raw/creditcard.csv').sample(1000)
current = simulate_drift(reference, drift_type='mean_shift', magnitude=1.0)

# Detect drift
result = detect_data_drift(reference, current)
print(f'Drift detected: {result[\"dataset_drift_detected\"]}')
print(f'Drifted features: {result[\"drifted_features\"]}')
"
```

## Detailed Usage

### Feature Engineering Pipeline

The pipeline transforms raw transaction data into 35 engineered features:

```python
from tabular_ml.features.pipeline import load_pipeline, fit_and_transform

# Load fitted pipeline (or fit new one)
pipeline = load_pipeline()  # Loads from artifacts/preprocessing_pipeline.joblib

# Transform new data
X_raw = pd.DataFrame(...)  # Raw transaction data
X_engineered = pipeline.transform(X_raw)  # 35 features
```

Transformations include:
- **Time features**: Cyclical encoding (`hour_sin`, `hour_cos`) from seconds
- **Amount transformation**: `log1p` + StandardScaler normalization
- **Interaction features**: Multiplications of top-correlated V-feature pairs
- **Original feature preservation**: All V1-V28 features retained

### Model Training & Tuning

```python
from tabular_ml.models.tuning import tune_xgboost
from tabular_ml.models.trainer import train_xgboost

# Hyperparameter tuning with Optuna (50 trials)
study = tune_xgboost(X_train, y_train, X_val, y_val)
best_params = study.best_params

# Train with optimal parameters
model, metrics = train_xgboost(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    test_data=(X_test, y_test),
    fixed_params=best_params
)

# Metrics include PR-AUC, ROC-AUC, F1, precision, recall
print(f"Test PR-AUC: {metrics['test_pr_auc']:.3f}")
```

### Ensemble Methods

```python
from tabular_ml.models.ensemble import StackingEnsemble

# Create stacking ensemble with 5-fold cross-validation
ensemble = StackingEnsemble()
ensemble.fit([xgboost_model, lightgbm_model, catboost_model], X_train, y_train)

# Predict with meta-learner (LogisticRegression)
y_pred_proba = ensemble.predict_proba(X_test)
```

### MLflow Experiment Tracking

All training runs are logged to MLflow:

```bash
# Start MLflow UI
mlflow ui --port 5000

# View experiments at http://localhost:5000
```

Each run includes:
- Hyperparameters and metrics
- PR curves and confusion matrices
- Model artifacts (serialized models)
- Feature importance plots

## Docker Deployment

### Build & Run API Container

```bash
# Build Docker image
docker build -t tabular-ml:latest .

# Run container
docker run -p 8000:8000 tabular-ml:latest

# Test containerized API
curl http://localhost:8000/health
```

### Docker Compose with MLflow

```bash
# Start both API and MLflow tracking server
docker-compose up -d

# Services:
# - API: http://localhost:8000
# - MLflow UI: http://localhost:5000

# View logs
docker-compose logs -f api
```

The Docker setup includes:
- Multi-stage builds for minimal image size (~1.8GB)
- Non-root user for security
- Health checks for container orchestration
- Volume persistence for MLflow artifacts

## Monitoring & Drift Detection

### Data Drift Monitoring

```python
from tabular_ml.monitoring import detect_data_drift, generate_drift_html_report

# Compare production vs reference data
drift_result = detect_data_drift(
    reference_data=training_data,
    current_data=production_data,
    threshold=0.05  # Significance level
)

# Generate HTML report
report_path = generate_drift_html_report(
    reference_data=training_data,
    current_data=production_data,
    output_dir="monitoring_reports",
    report_filename="weekly_drift_report.html"
)
```

### Drift Simulation for Testing

```python
from tabular_ml.monitoring import simulate_drift

# Simulate different types of drift
drifted = simulate_drift(data, drift_type="mean_shift", magnitude=1.0)
drifted = simulate_drift(data, drift_type="scale_change", magnitude=0.5)
drifted = simulate_drift(data, drift_type="corruption", magnitude=0.2)
drifted = simulate_drift(data, drift_type="missing", magnitude=0.1)
```

## Testing

The project includes 69 comprehensive tests:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_features.py -v
pytest tests/test_models.py -v
pytest tests/test_api.py -v
pytest tests/test_monitoring.py -v

# Run with coverage
pytest --cov=src/tabular_ml --cov-report=html
```

## Reproducibility

### Environment Setup

```bash
# Exact environment replication
pip freeze > requirements.frozen.txt
pip install -r requirements.frozen.txt

# Or using conda
conda env create -f environment.yml  # If provided
```

### Data Versioning

The raw dataset is excluded from git (in `.gitignore`). To reproduce:

1. Download from Kaggle: `kaggle datasets download -d mlg-ulb/creditcardfraud`
2. Place in `data/raw/creditcard.csv`
3. All subsequent artifacts are generated from this source

### Configuration Management

All pipeline parameters are centralized in `configs/default.yaml`:

```yaml
data:
  raw_path: "data/raw/creditcard.csv"
  target_column: "Class"

split:
  test_size: 0.2
  val_size: 0.2
  random_state: 42
  stratify: true

features:
  time_period_seconds: 86400
  amount_log_transform: true
  amount_standardize: true
  interaction_pairs:
    - ["V14", "V17"]
    - ["V12", "V14"]
    - ["V10", "V17"]
    - ["V4", "V11"]
```

## Development

### Adding New Features

1. **New transformers**: Add to `src/tabular_ml/features/engineering.py`
2. **New models**: Add to `src/tabular_ml/models/trainer.py` and `tuning.py`
3. **New API endpoints**: Add to `src/tabular_ml/api/app.py` and `schemas.py`
4. **New monitoring metrics**: Add to `src/tabular_ml/monitoring/drift.py`

### Code Quality

```bash
# Type checking (if mypy configuration added)
mypy src/

# Linting (if ruff/flake8 configuration added)
ruff check src/
```

## Performance

- **Training time**: ~10 minutes for full pipeline (3 models × 50 Optuna trials)
- **Inference latency**: <10ms per transaction
- **Memory usage**: ~2GB for training, ~500MB for inference
- **Container size**: 1.8GB (includes all dependencies)

## Limitations & Future Work

- **Dataset**: Limited to 284k transactions; real-world systems handle millions
- **Features**: PCA-transformed features (V1-V28) limit interpretability
- **Real-time**: Batch inference only; streaming inference not implemented
- **Model retraining**: Automated retraining pipeline not implemented
- **Advanced monitoring**: Prediction drift and concept drift not monitored

Potential enhancements:
- Real-time streaming with Kafka/WebSockets
- Automated model retraining pipeline
- Advanced monitoring (prediction drift, concept drift)
- A/B testing framework for model deployment
- Feature store integration for consistent feature engineering

## License

MIT License. See LICENSE file for details.

## Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Built with Python's data science ecosystem
- Inspired by production MLOps best practices