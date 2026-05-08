# ✈️ AeroPredict — Predictive Maintenance AI

> An end-to-end MLOps pipeline that predicts **Remaining Useful Life (RUL)** of aircraft engines using XGBoost, with full CI/CD automation via GitHub Actions, DVC, and DagsHub.

![CI/CD](https://github.com/SAIF-BT/Predictive_maintenance/actions/workflows/mlops.yaml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![DVC](https://img.shields.io/badge/Data-DVC-purple)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Configuration](#-configuration)
- [Experiment Tracking](#-experiment-tracking)

---

## 🔍 Overview

AeroPredict is a production-grade **predictive maintenance system** for jet engines. It uses sensor data from NASA's CMAPS benchmark dataset to predict how many flight cycles remain before an engine requires maintenance or fails.

The system is built with a complete MLOps workflow:

- **Automated data ingestion** from Kaggle on every pipeline run
- **Reproducible pipeline** with DVC — every stage tracked and versioned
- **Experiment tracking** with MLflow on DagsHub
- **CI/CD** via GitHub Actions — trains and deploys on every push to main
- **Interactive dashboard** built with Streamlit for real-time RUL prediction

---

## 🚀 Live Demo

> **[Launch AeroPredict Dashboard →](https://predictivemaintenances.streamlit.app/)**

The dashboard lets you:
- Select any engine from the fleet
- Scrub through flight cycles using a time slider
- See predicted RUL, health status, and degradation trend in real time

---

## 📊 Model Performance

Trained on NASA CMAPS **FD002** dataset — the hardest subset, with 6 operating conditions and 2 fault modes.

| Metric | Value | Meaning |
|--------|-------|---------|
| **RMSE** | 19.05 cycles | Penalised average error |
| **MAE** | 13.75 cycles | Plain average error |

### Benchmark comparison

| Model | RMSE on FD002 |
|-------|---------------|
| Linear Regression (baseline) | ~38 |
| Basic LSTM (no tuning) | ~24 |
| **AeroPredict XGBoost** | **19.05** |
| State of the art (research papers) | ~12–15 |

> AeroPredict outperforms basic deep learning models with a well-tuned gradient boosting approach.

---

## 📁 Project Structure

```
Predictive_maintenance/
│
├── src/
│   ├── data_ingestion.py     # Downloads dataset from Kaggle
│   ├── preprocess.py         # RUL computation + regime-based scaling
│   ├── features.py           # Rolling window feature engineering
│   ├── train.py              # XGBoost training + metric logging
│   ├── visualization.py      # RUL prediction + feature importance plots
│   ├── predict.py            # Single engine inference utility
│   ├── app.py                # Streamlit dashboard
│   └── dvc_setup.py          # Model download helper for Streamlit Cloud
│
├── data/
│   ├── raw/                  # Downloaded from Kaggle (DVC tracked)
│   └── processed/            # Preprocessed + feature CSV files (DVC tracked)
│
├── models/
│   ├── model_xgb.json        # Trained XGBoost model (DVC tracked)
│   └── scalers.pkl           # Per-regime StandardScalers (DVC tracked)
│
├── plots/
│   ├── engine_prediction.png # RUL prediction plot
│   └── feature_importance.png # Top 15 features chart
│
├── .github/
│   └── workflows/
│       └── mlops.yaml        # GitHub Actions CI/CD pipeline
│
├── dvc.yaml                  # Pipeline stage definitions
├── dvc.lock                  # Reproducibility lock file
├── params.yaml               # Hyperparameters and config
├── metrics.json              # Latest model metrics
└── requirements.txt          # Python dependencies
```

---

## 🏗️ Pipeline Architecture

```
NASA CMAPS Dataset (Kaggle)
         │
         ▼
  data_ingestion.py       ← Downloads raw .txt sensor files
         │
         ▼
    preprocess.py          ← Computes piecewise RUL (clipped at 125)
                            ← Groups rows into flight regimes
                            ← StandardScaler per regime → scalers.pkl
         │
         ▼
     features.py           ← Rolling 10-cycle window per engine
                            ← Adds s_N_mean + s_N_std (63 features total)
         │
         ▼
      train.py             ← 80/20 group split (by engine unit)
                            ← XGBoost regressor → model_xgb.json
                            ← Logs RMSE + MAE → metrics.json
         │
         ▼
   visualization.py        ← RUL prediction plot per engine
                            ← Feature importance chart
         │
         ▼
    DagsHub (DVC remote)   ← All artifacts versioned and stored
         │
         ▼
       app.py              ← Pulls model → real-time prediction
```

---

## 🧠 How It Works

### 1. Data — NASA CMAPS FD002

The dataset contains multivariate time-series sensor readings from turbofan jet engines. Each engine runs from a healthy state until failure. The task is to predict how many cycles remain.

- **260 engines**in training set
- **21 sensors** per cycle (temperature, pressure, speed, vibration)
- **3 operational settings** defining the flight regime
- **Target**: Remaining Useful Life (RUL), clipped at 125 cycles

### 2. Regime-based Preprocessing

FD002 has 6 different operating conditions (altitude + throttle combinations). Sensors behave differently in each regime, so each is scaled independently:

```python
# Separate StandardScaler fitted per regime
for regime in df['regime'].unique():
    mask = df['regime'] == regime
    s = StandardScaler()
    df.loc[mask, sensors] = s.fit_transform(df.loc[mask, sensors])
    scalers[regime] = s
```

### 3. Feature Engineering

Raw sensor readings are augmented with rolling statistics over a 10-cycle window, giving the model trend information rather than just snapshots:

- `s_N_mean` — rolling average of each sensor
- `s_N_std` — rolling standard deviation of each sensor

This creates **63 features** from 21 raw sensors.

### 4. Training Strategy

Engines are kept together in the train/validation split to prevent data leakage:

```python
train_units = units[:int(len(units) * 0.8)]   # 208 engines
val_units   = units[int(len(units) * 0.8):]   # 52 engines
```

### 5. Real-Time Prediction

When a user selects an engine and cycle in the Streamlit app:

1. The app loads the feature row for that cycle
2. Calls `model.predict(X)` — runs in milliseconds
3. Displays predicted RUL and health status

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | XGBoost |
| Pipeline orchestration | DVC |
| Experiment tracking | MLflow + DagsHub |
| CI/CD | GitHub Actions |
| Data storage | DagsHub (S3-compatible) |
| Dashboard | Streamlit |
| Visualizations | Plotly + Matplotlib |
| Data source | Kaggle API |
| Language | Python 3.9 |

---

## ⚡ Getting Started

### Prerequisites

- Python 3.9+
- Kaggle account with API token
- DagsHub account

### 1. Clone the repo

```bash
git clone https://github.com/SAIF-BT/Predictive_maintenance.git
cd Predictive_maintenance
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export KAGGLE_API_TOKEN=your_kaggle_token
export DAGSHUB_TOKEN=your_dagshub_token
```

### 4. Configure DVC remote

```bash
dvc remote add -d origin https://dagshub.com/SAIF-BT/Predictive_maintenance.dvc
dvc remote modify origin auth basic
dvc remote modify origin user SAIF-BT
dvc remote modify origin password $DAGSHUB_TOKEN
```

### 5. Run the full pipeline

```bash
dvc repro
```

This runs all 5 stages in order: `ingest → preprocess → features → train → visualize`

### 6. Launch the dashboard

```bash
streamlit run src/app.py
```

### Run a single engine prediction

```bash
python src/predict.py
```

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow runs automatically on:

| Trigger | When |
|---------|------|
| Push to `main` | Every code change |
| Scheduled | Every Monday at 6am UTC |
| Manual | "Run workflow" button in Actions tab |

### Pipeline stages in CI

```yaml
Install dependencies (pinned versions, cached)
       ↓
Configure DVC remote (DagsHub auth)
       ↓
Restore DVC cache (dvc pull)
       ↓
Run full pipeline (dvc repro)
       ↓
Push artifacts (dvc push)
       ↓
Report metrics (RMSE + MAE in Actions summary)
```

### Required GitHub Secrets

Go to your repo → **Settings** → **Secrets and variables** → **Actions** and add:

| Secret | Description |
|--------|-------------|
| `KAGGLE_API_TOKEN` | From kaggle.com → Settings → API |
| `DAGSHUB_TOKEN` | From dagshub.com → Settings → Tokens |

---

## ⚙️ Configuration

All hyperparameters are in `params.yaml`. Change any value and push — the pipeline retrains automatically.

```yaml
data:
  dataset: FD002          # NASA CMAPS subset (FD001-FD004)

train:
  n_estimators: 200       # Number of trees
  learning_rate: 0.05     # Step size per tree
  max_depth: 6            # Tree depth
  random_state: 42        # Reproducibility seed
```

### To experiment with hyperparameters

```bash
# Edit params.yaml locally
vim params.yaml

# Run pipeline
dvc repro

# Compare metrics
dvc metrics show
dvc metrics diff
```

---

## 📈 Experiment Tracking

All runs are tracked on DagsHub with MLflow:

**[View Experiment Dashboard →](https://dagshub.com/SAIF-BT/Predictive_maintenance.mlflow)**

Each run logs:
- RMSE and MAE on validation set
- All hyperparameters from `params.yaml`
- Model artifact

---

## 📄 License

This project uses the [NASA CMAPS Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) for research purposes.

---

<p align="center">Built with ❤️ by <a href="https://github.com/SAIF-BT">SAIF-BT</a></p>
