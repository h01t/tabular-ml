# Tabular ML — Task Tracker

## Phase 0: Project Scaffolding ✓
- [x] Initialize git repo
- [x] Create project directory structure (src layout)
- [x] Create requirements.txt
- [x] Create .gitignore
- [x] Download Kaggle fraud detection dataset
- [x] Set up tasks/todo.md and tasks/lessons.md
- [x] Create initial README.md
- [x] Install dependencies
- [x] Initial git commit

## Phase 1: EDA ✓
- [x] Create notebooks/01_eda.ipynb
- [x] Document dataset shape, dtypes, missing values, duplicates
- [x] Analyze class distribution
- [x] Distribution plots (Amount, Time, key V-features)
- [x] Correlation analysis (heatmap, bar chart, top-15 features)
- [x] Outlier analysis
- [x] Fraud deep dive (statistical comparison)
- [x] Summary of findings & modeling implications
- [x] Verified: notebook executes end-to-end, 6 plots generated

## Phase 2: Feature Engineering ✓
- [x] Data loading module (src/tabular_ml/data/loader.py)
- [x] Stratified train/val/test split (80/16/20 with class ratio preserved)
- [x] TimeFeatureExtractor — cyclical hour_sin/hour_cos from Time column
- [x] AmountTransformer — log1p + StandardScaler on Amount
- [x] InteractionFeatureCreator — V14*V17, V12*V14, V10*V17, V4*V11
- [x] build_feature_pipeline() — composable sklearn Pipeline
- [x] Pipeline orchestration (fit_and_transform, save/load)
- [x] YAML config (configs/default.yaml)
- [x] 22 unit tests — all passing, zero warnings
- [x] Verified: 30 raw features → 35 engineered features, pipeline serialized

## Phase 3: Model Training & Tuning ✓
- [x] Evaluation module (PR-AUC, ROC-AUC, F1, precision, recall, optimal threshold)
- [x] Trainer with MLflow logging (params, metrics, PR curves, confusion matrices)
- [x] Optuna tuning module (50 trials per model, PR-AUC optimization)
- [x] XGBoost — PR-AUC: 0.848 val / 0.867 test (best individual)
- [x] LightGBM — PR-AUC: 0.833 val / 0.864 test
- [x] CatBoost — PR-AUC: 0.785 val / 0.838 test
- [x] Class imbalance handled (scale_pos_weight / auto_class_weights)
- [x] 3 MLflow runs logged with full artifacts
- [x] 39 unit tests passing
- [x] Results saved to artifacts/training_results.json

## Phase 4: Ensemble ✓
- [x] StackingEnsemble — 5-fold OOF predictions + LogisticRegression meta-learner
- [x] BlendingEnsemble — optimized weight grid search on validation set
- [x] Comparison: XGBoost best on PR-AUC (0.867), Stacking best on recall (0.847)
- [x] 2 MLflow runs logged (stacking + blending)
- [x] 12 ensemble tests + 51 total tests passing
- [x] Results saved to artifacts/ensemble_results.json

## Phase 5: FastAPI Inference Service ✓
- [x] Pydantic schemas (TransactionFeatures, PredictionResponse, BatchPredictionRequest/Response, HealthResponse)
- [x] FastAPI app with GET /health, POST /predict, POST /predict/batch
- [x] Loads preprocessing pipeline + XGBoost model at startup via lifespan
- [x] Input validation (Amount >= 0, all 30 features required, batch max 10k)
- [x] Response includes fraud_probability, is_fraud, threshold, model metadata
- [x] 9 API tests (health, single predict, batch, validation errors, consistency)
- [x] 60 total tests passing

## Phase 6: Docker ✓
- [x] Multi-stage Dockerfile (builder + slim runtime, non-root user, healthcheck)
- [x] requirements-docker.txt (minimal inference deps only)
- [x] .dockerignore (excludes data, notebooks, tests, mlruns, unused models)
- [x] docker-compose.yml (API service + MLflow tracking server)
- [x] Image built: tabular-ml:latest (1.8GB)
- [x] Verified: /health, /predict, /predict/batch all working from container

## Phase 7: Monitoring ✓
- [x] Implement drift detection with Evidently
- [x] Simulate drift
- [x] Generate dashboard/report

## Phase 8: Documentation & Polish ✓
- [x] Clean README with architecture, setup, results
- [x] Reproducibility instructions
- [x] Final code review
