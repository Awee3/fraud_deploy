# Fraud Detection MLOps Platform

A containerized fraud-detection platform built with FastAPI, machine-learning models, SQLite, statistical drift monitoring, traffic simulation, and n8n automation.

## Features

- Real-time fraud prediction through a REST API
- XGBoost and scikit-learn model support
- Model hot-reloading without restarting the API
- Prediction logging in SQLite
- Request latency and error-rate monitoring
- Feature drift detection using the Kolmogorov-Smirnov test
- Prediction drift detection using Fisher's Exact Test or Chi-Squared Test
- Automated monitoring workflows through n8n
- Automated model deployment workflows through n8n
- Deployment lead-time tracking
- Synthetic traffic and drift scenario generation

## Architecture

```text
Client / Simulator
        |
        v
FastAPI Fraud Detection API (:5000)
        |
        +--------------------+
        |                    |
        v                    v
Model Files (*.pkl)     SQLite Database
                         - Predictions
                         - Request Metrics
                         - Deployment Metrics
                                |
                                v
                         Drift Monitoring
                         - Feature Drift
                         - Prediction Drift
                         - Operational Metrics
                                |
                                v
                         n8n Automation
                         - Alerts
                         - Deployment
                         - Smoke Tests
```

## Project Structure

```text
fraud_deploy/
├── api/
│   ├── database.py
│   ├── dockerfile
│   ├── main.py
│   ├── monitoring.py
│   └── requirements.txt
├── logs/
│   ├── prediction_logs.csv
│   └── predictions.db
├── models/
│   └── *.pkl
├── monitoring_config/
│   ├── Fraud Detection.json
│   ├── Fraud Detection V2.json
│   ├── n8n_workflow.json
│   ├── static_allbaseline.json
│   └── static_baseline.json
├── simulation/
│   ├── simulate_traffic.py
│   ├── test_data_sample.csv
│   └── *.csv
├── tests/
│   ├── scenario1_hotreload/
│   ├── scenario2_drift/
│   ├── scenario3_efficiency/
│   ├── fixtures/
│   └── utils/
├── docker-compose.yml
├── hasil_drift_neg.json
├── hasil_prior_shift.json
└── test.py
```

## Main Components

### Fraud Detection API

The API is implemented in [`api/main.py`](api/main.py).

It:

1. Loads a machine-learning model during startup.
2. Accepts transaction data through `/predict`.
3. Generates a fraud prediction and probability.
4. Stores prediction results in SQLite.
5. Records request latency and HTTP errors.
6. Supports model hot-reloading through `/reload`.
7. Runs drift monitoring through `/run-monitoring`.

### Database Layer

The database logic is implemented in [`api/database.py`](api/database.py).

The SQLite database stores:

- Prediction records
- Request metrics
- Deployment metrics

The database schema is created automatically when the API starts.

### Monitoring

The monitoring system is implemented in [`api/monitoring.py`](api/monitoring.py).

It evaluates:

- Feature distribution drift
- Fraud prediction-rate drift
- API latency
- API error rate
- Deployment information

### Traffic Simulation

The traffic simulator is implemented in [`simulation/simulate_traffic.py`](simulation/simulate_traffic.py).

It reads transaction data from CSV files and sends each row to the prediction API.

Available datasets include:

- `simulasi_normal.csv`
- `simulasi_produksi.csv`
- `simulasi_drift.csv`
- `simulasi_drift_2k.csv`
- `simulasi_drift_500.csv`
- `simulasi_drift_neg.csv`
- `simulasi_prior_shift.csv`
- `test_data_sample.csv`

### n8n Automation

The workflow is stored in [`monitoring_config/n8n_workflow.json`](monitoring_config/n8n_workflow.json).

It supports:

- Scheduled monitoring
- Drift notifications
- GitHub-triggered model deployment
- Downloading model artifacts
- Saving models to the shared model volume
- Calling the API reload endpoint
- Running deployment smoke tests
- Recording deployment lead time
- Sending deployment success and failure notifications

## Feature Contract

The API expects the following 29 features:

```text
V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
V21, V22, V23, V24, V25, V26, V27, V28, Amount
```

`Class` and `Time` are not part of the prediction contract.

A model may be stored as a bundle:

```python
{
    "model": model,
    "scaler": scaler,
    "top_features": [...]
}
```

The API also supports standalone estimators.

If a scaler is included, the API scales `Amount` before prediction and logging.

## Requirements

### Docker

- Docker
- Docker Compose

### Local Development

- Python 3.11 or compatible
- FastAPI
- Uvicorn
- pandas
- scikit-learn 1.7.2
- joblib
- NumPy
- SciPy
- XGBoost 3.2.0
- httpx

Python dependencies are listed in [`api/requirements.txt`](api/requirements.txt).

## Running with Docker Compose

From the project root:

```bash
docker compose up --build
```

Run in the background:

```bash
docker compose up --build -d
```

Stop the services:

```bash
docker compose down
```

Available services:

| Service | Address |
|---|---|
| Fraud API | `http://localhost:5000` |
| n8n | `http://localhost:5678` |

The Docker Compose configuration is defined in [`docker-compose.yml`](docker-compose.yml).

## Environment Variables

The API supports the following variables:

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `logs/predictions.db` | SQLite database path |
| `MODEL_DIR` | `models` | Directory containing model files |
| `BASELINE_PATH` | `monitoring_config/static_allbaseline.json` | Drift baseline path |

The Docker configuration uses:

```text
DB_PATH=/app/logs/predictions.db
MODEL_DIR=/app/models
BASELINE_PATH=/app/monitoring_config/static_allbaseline.json
```

The simulator supports:

| Variable | Default | Description |
|---|---|---|
| `API_URL` | `http://localhost:5000/predict` | Prediction endpoint |
| `DELAY_SECONDS` | `0.5` | Approximate delay between requests |
| `REQUEST_TIMEOUT` | `10` | Request timeout in seconds |
| `DROP_COLUMNS` | `Class` | Comma-separated columns removed before prediction |

## API Endpoints

### `GET /health`

Returns API, model, database, operational, and deployment status.

```bash
curl http://localhost:5000/health
```

### `POST /predict`

Generates a fraud prediction.

Example payload:

```json
{
  "V1": 0.0,
  "V2": 0.0,
  "V3": 0.0,
  "V4": 0.0,
  "V5": 0.0,
  "V6": 0.0,
  "V7": 0.0,
  "V8": 0.0,
  "V9": 0.0,
  "V10": 0.0,
  "V11": 0.0,
  "V12": 0.0,
  "V13": 0.0,
  "V14": 0.0,
  "V15": 0.0,
  "V16": 0.0,
  "V17": 0.0,
  "V18": 0.0,
  "V19": 0.0,
  "V20": 0.0,
  "V21": 0.0,
  "V22": 0.0,
  "V23": 0.0,
  "V24": 0.0,
  "V25": 0.0,
  "V26": 0.0,
  "V27": 0.0,
  "V28": 0.0,
  "Amount": 100.0
}
```

Example request:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @payload.json
```

Example response:

```json
{
  "record_id": 1,
  "prediction": 0,
  "probability": 0.0021,
  "status": "logged"
}
```

### `GET /logs`

Returns recent prediction logs.

Parameters:

- `limit`: Number of records
- Minimum: `1`
- Maximum: `5000`
- Default: `500`

```bash
curl "http://localhost:5000/logs?limit=100"
```

### `POST /reload`

Reloads the default model:

```bash
curl -X POST http://localhost:5000/reload
```

Reloads a specific model:

```bash
curl -X POST "http://localhost:5000/reload?model_name=xgboost_model_v2.pkl"
```

The API validates the candidate model before activation by checking that:

- The file exists.
- The model can be loaded.
- The model supports the API feature contract.
- A dummy prediction succeeds.

If validation fails, the currently active model remains unchanged.

### `POST /run-monitoring`

Runs feature-drift, prediction-drift, and operational monitoring.

Parameters:

- `limit`: Recent prediction logs, between `100` and `5000`
- `window_minutes`: Operational metric window, between `10` and `1440`

```bash
curl -X POST \
  "http://localhost:5000/run-monitoring?limit=1000&window_minutes=60"
```

### `GET /metrics`

Returns request count, error rate, latency percentiles, and endpoint breakdown.

```bash
curl "http://localhost:5000/metrics?window_minutes=60"
```

### `POST /metrics/deployment`

Records deployment information.

Required fields:

```json
{
  "model_name": "xgboost_model_v2.pkl",
  "lead_time_seconds": 4.25,
  "status": "success"
}
```

Optional field:

```json
{
  "triggered_by": "github_webhook"
}
```

Example:

```bash
curl -X POST http://localhost:5000/metrics/deployment \
  -H "Content-Type: application/json" \
  -d "{\"model_name\":\"xgboost_model_v2.pkl\",\"lead_time_seconds\":4.25,\"status\":\"success\"}"
```

### `GET /metrics/deployment`

Returns recent deployment history.

```bash
curl "http://localhost:5000/metrics/deployment?limit=10"
```

### API Documentation

FastAPI provides:

- Swagger UI: `http://localhost:5000/docs`
- OpenAPI JSON: `http://localhost:5000/openapi.json`

## Running the Traffic Simulator

Start the API first, then run:

```bash
python simulation\simulate_traffic.py
```

This uses `test_data_sample.csv` by default.

Use another dataset:

```bash
python simulation\simulate_traffic.py simulasi_normal.csv
```

Example with custom settings:

```powershell
$env:API_URL="http://localhost:5000/predict"
$env:DELAY_SECONDS="0.2"
python simulation\simulate_traffic.py simulasi_produksi.csv
```

The simulator reports:

- Number of transactions sent
- Successful predictions
- Failed requests
- Prediction results
- Fraud probabilities
- Database record IDs

## Drift Monitoring

The monitoring system evaluates 29 features:

```text
V1 through V28 and Amount
```

The baseline is loaded from:

```text
monitoring_config/static_allbaseline.json
```

Monitoring requires:

- At least 30 production samples per feature
- At least 30 prediction samples for prediction-distribution drift

### Feature Drift

Feature drift uses the two-sample Kolmogorov-Smirnov test.

### Tier 1 Features

Tier-1 features are the most important model features.

- Significance threshold: `p < 0.01`
- Drift in one Tier-1 feature triggers `CRITICAL`

The Tier-1 feature list is read from the baseline metadata.

### Tier 2 Features

Tier-2 features are the remaining features.

- Significance threshold: `p < 0.05`
- At least five simultaneously drifting Tier-2 features trigger `WARNING`

### Prediction Drift

Prediction drift compares the observed fraud rate with the baseline fraud rate.

- Fisher's Exact Test is used when the expected fraud count is small.
- Chi-Squared Test is used when the expected fraud count is sufficiently large.
- The significance threshold is `p < 0.01`.

### Alert Levels

| Alert Level | Meaning |
|---|---|
| `NONE` | No statistically significant drift |
| `WARNING` | Tier-2 drift threshold or prediction drift detected |
| `CRITICAL` | At least one Tier-1 feature has significant drift |
| `ERROR` | Monitoring could not produce a trustworthy result |

Monitoring responses include:

- Overall status
- Alert level
- Trigger reason
- Number of samples
- Monitored features
- Skipped features
- Drifted features
- Tier-1 and Tier-2 results
- Prediction-drift details
- Operational metrics
- Last deployment information

## Database

The SQLite database is stored at:

```text
logs/predictions.db
```

### `predictions`

Stores:

- Timestamp
- 29 input features
- Prediction
- Prediction probability

### `request_metrics`

Stores:

- Timestamp
- Endpoint
- HTTP method
- Status code
- Request latency
- Error indicator

### `deployment_metrics`

Stores:

- Timestamp
- Model name
- Deployment lead time
- Deployment status
- Trigger source

The database is created automatically when the API starts.

## Model Management

Model files are stored in:

```text
models/
```

The default model is:

```text
models/xgboost_model.pkl
```

Reload a model with:

```bash
curl -X POST \
  "http://localhost:5000/reload?model_name=xgboost_model.pkl"
```

Only models that pass validation are activated.

Models must be compatible with the 29-feature API contract.

## n8n Workflow

Import the workflow from:

```text
monitoring_config/n8n_workflow.json
```

### Scheduled Monitoring Flow

1. A schedule trigger starts monitoring.
2. n8n calls `/run-monitoring`.
3. The workflow evaluates the monitoring status.
4. Drift alerts are sent when drift is detected.
5. No alert is sent when the system is healthy.

### Automated Deployment Flow

1. A deployment webhook receives a GitHub payload.
2. n8n searches for a changed `.pkl` file.
3. The model is downloaded.
4. The model is saved to the shared model volume.
5. n8n calls `/reload`.
6. The API validates and activates the model.
7. A health check is performed.
8. Deployment lead time is calculated.
9. Deployment metrics are recorded.
10. A success or failure notification is sent.

Before enabling the workflow:

- Configure GitHub authentication.
- Configure Telegram credentials.
- Review repository URLs.
- Review webhook settings.
- Verify that n8n and the API share the same Docker network.
- Store secrets in n8n credentials or environment variables.
- Do not commit access tokens or private credentials.

## Local Development Without Docker

Create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r api\requirements.txt
```

Start the API:

```powershell
cd api
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

Configure paths if necessary:

```powershell
$env:MODEL_DIR="D:\fraud_deploy\models"
$env:DB_PATH="D:\fraud_deploy\logs\predictions.db"
$env:BASELINE_PATH="D:\fraud_deploy\monitoring_config\static_allbaseline.json"
```

## Testing and Experiment Scenarios

### Scenario 1: Model Hot Reloading

Located in:

```text
tests/scenario1_hotreload/
```

Used to evaluate:

- Model reload behavior
- Reload triggers
- Prediction continuity
- Load generation
- Reload analysis

### Scenario 2: Drift Detection

Located in:

```text
tests/scenario2_drift/
```

Used to evaluate:

- Feature drift
- Prior shift
- Detection latency
- Repeated monitoring behavior

### Scenario 3: Efficiency

Located in:

```text
tests/scenario3_efficiency/
```

Used to evaluate:

- API latency
- Automated deployment lead time
- Manual deployment lead time

### Utilities

Located in:

```text
tests/utils/
```

Utilities include state reset and drift-feature discovery helpers.

## Example End-to-End Workflow

Start the services:

```bash
docker compose up --build -d
```

Check the API:

```bash
curl http://localhost:5000/health
```

Generate traffic:

```bash
python simulation\simulate_traffic.py simulasi_normal.csv
```

Check prediction logs:

```bash
curl "http://localhost:5000/logs?limit=10"
```

Run monitoring:

```bash
curl -X POST \
  "http://localhost:5000/run-monitoring?limit=1000&window_minutes=60"
```

Check operational metrics:

```bash
curl "http://localhost:5000/metrics?window_minutes=60"
```

Reload another model:

```bash
curl -X POST \
  "http://localhost:5000/reload?model_name=xgboost_model_v2.pkl"
```

Verify the active model:

```bash
curl http://localhost:5000/health
```

## Troubleshooting

### Model Is Not Loaded

Check that:

- The model file exists in `models/`.
- `MODEL_DIR` points to the correct directory.
- The model can be loaded with `joblib`.
- The model supports the 29-feature API contract.

### Prediction Returns HTTP 503

The API does not have a valid loaded model. Check the API logs and `/health`.

### Prediction Returns HTTP 400

Check that:

- The request body is a non-empty JSON object.
- Required features are present.
- Values are numeric.
- The model accepts the provided features.

### Monitoring Returns `ERROR`

Common causes:

- No prediction logs exist.
- There are not enough samples.
- The baseline file is missing.
- The baseline does not define `meta.tier1_features`.
- Baseline feature names do not match production log feature names.

### n8n Cannot Reach the API

From inside the n8n container, use:

```text
http://fraud-api-container:5000
```

Do not use `localhost` from inside n8n because it refers to the n8n container itself.

### Database Data Disappears

Ensure the logs directory is mounted into the API container:

```yaml
- ./logs:/app/logs
```

## Operational Notes

- Review all n8n integration settings before production use.
- Store credentials securely.
- The API does not currently provide authentication.
- Place the API behind an authenticated gateway or private network in production.
- Model artifacts should be versioned and validated before deployment.
- Drift alerts are statistical signals and require investigation.
- A drift alert may require checking data pipelines, feature transformations, model performance, and retraining requirements.
- Add a license file before distributing the project publicly.The README content has been provided in a copy-paste-ready format with no surrounding explanation.

**Task completed:**

Provided the complete README content in Markdown format for direct copying into readme.md.