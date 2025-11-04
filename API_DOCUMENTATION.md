# 📚 API Documentation - Road Traffic Accident Classification

<div align="center">

![Version](https://img.shields.io/badge/API%20Version-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?logo=python)
![Status](https://img.shields.io/badge/Status-Stable-success)

**Complete technical reference for the RTA Classification system**

[🏠 Home](README.md) • [📋 Project Plan](PROJECT_PLAN.md) • [🗺️ Roadmap](ROADMAP.md)

</div>

---

## 📑 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Modules](#core-modules)
- [Data Pipeline](#data-pipeline)
- [Model Training](#model-training)
- [Prediction API](#prediction-api)
- [Web Application](#web-application)
- [Code Examples](#code-examples)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Performance Optimization](#performance-optimization)
- [Best Practices](#best-practices)

---

## 🎯 Overview

### System Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLIENT LAYER                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │   CLI    │  │ Web App  │  │ REST API │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
└───────┼─────────────┼─────────────┼───────────────┘
        │             │             │
┌───────▼─────────────▼─────────────▼───────────────┐
│              APPLICATION LAYER                     │
│  ┌──────────────────────────────────────────┐     │
│  │      Preprocessing Pipeline              │     │
│  └──────────┬───────────────────────────────┘     │
│             │                                      │
│  ┌──────────▼───────────────────────────────┐     │
│  │      Model Inference Engine              │     │
│  └──────────┬───────────────────────────────┘     │
│             │                                      │
│  ┌──────────▼───────────────────────────────┐     │
│  │      SHAP Explainability Module          │     │
│  └──────────────────────────────────────────┘     │
└────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────┐
│              DATA LAYER                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Raw Data │  │ Processed│  │  Models  │     │
│  └──────────┘  └──────────┘  └──────────┘     │
└─────────────────────────────────────────────────┘
```

### Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `get_dataset.py` | Data acquisition | `download_dataset()` |
| `preprocessing.py` | Data preprocessing | `preprocess_data()`, `knn_impute()` |
| `oversampler.py` | Oversampling | `apply_oversampling()` |
| `undersampler.py` | Undersampling | `apply_undersampling()` |
| `combined_sampling.py` | Hybrid sampling | `apply_combined_sampling()` |
| `base_model_evaluation.py` | Model training | `train_evaluate_model()` |
| `feature_selection.py` | Feature engineering | `extract_feature_importance()` |
| `hp_tunnig_et.py` | Hyperparameter tuning | `bayesian_optimization()` |
| `train_and_evaluate_model.py` | Final training | `train_final_model()` |
| `XAI_SHAP_v2.py` | Explainability | `compute_shap_values()` |
| `streamlit_app.py` | Web interface | `main()`, `predict()` |

---

## 💻 Installation

### Prerequisites

```bash
# System requirements
Python >= 3.8
pip >= 21.0
virtualenv (recommended)

# Optional
Docker >= 20.0 (for containerization)
Git >= 2.0 (for version control)
```

### Standard Installation

```bash
# Clone repository
git clone https://github.com/zubairashfaque/Road-Traffic-Accident-Classification.git
cd Road-Traffic-Accident-Classification

# Create virtual environment
python -m venv rta_env
source rta_env/bin/activate  # On Windows: rta_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Docker Installation (Future)

```bash
# Build Docker image
docker build -t rta-classifier .

# Run container
docker run -p 8501:8501 rta-classifier
```

---

## 🔧 Core Modules

### 1. Data Acquisition (`get_dataset.py`)

#### Overview
Downloads RTA dataset from Kaggle using the Kaggle API.

#### Functions

##### `setup_kaggle_credentials()`
```python
def setup_kaggle_credentials():
    """
    Setup Kaggle API credentials from environment variables.

    Environment Variables:
        KAGGLE_USERNAME: Your Kaggle username
        KAGGLE_KEY: Your Kaggle API key

    Raises:
        EnvironmentError: If credentials are not found
    """
```

##### `download_dataset(dataset_name, download_dir)`
```python
def download_dataset(
    dataset_name: str,
    download_dir: str = "data/raw"
) -> bool:
    """
    Download dataset from Kaggle and extract files.

    Args:
        dataset_name (str): Kaggle dataset identifier
            e.g., "saurabhshahane/road-traffic-accidents"
        download_dir (str): Directory to save downloaded files

    Returns:
        bool: True if download successful, False otherwise

    Example:
        >>> download_dataset("saurabhshahane/road-traffic-accidents")
        Downloading dataset...
        Dataset downloaded successfully!
        True
    """
```

#### Usage Example

```python
from src.get_dataset import download_dataset

# Download dataset
success = download_dataset(
    dataset_name="saurabhshahane/road-traffic-accidents",
    download_dir="data/raw"
)

if success:
    print("Dataset ready for preprocessing!")
```

#### CLI Usage

```bash
# Run directly
python src/get_dataset.py

# Or using Makefile
make download_data
```

---

### 2. Preprocessing (`preprocessing.py`)

#### Overview
Comprehensive data preprocessing pipeline including imputation, encoding, and feature engineering.

#### Functions

##### `load_data(file_path)`
```python
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV data into pandas DataFrame.

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataframe

    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
```

##### `drop_unnecessary_columns(df, columns_to_drop)`
```python
def drop_unnecessary_columns(
    df: pd.DataFrame,
    columns_to_drop: List[str]
) -> pd.DataFrame:
    """
    Remove specified columns from dataframe.

    Args:
        df: Input dataframe
        columns_to_drop: List of column names to remove

    Returns:
        pd.DataFrame: Dataframe with columns removed
    """
```

##### `create_time_features(df, time_column)`
```python
def create_time_features(
    df: pd.DataFrame,
    time_column: str = 'Time'
) -> pd.DataFrame:
    """
    Extract temporal features from time column.

    Creates:
        - hour: Hour of day (0-23)
        - minute: Minute of hour (0-59)
        - session: Time period category
            - morning: 6-12
            - afternoon: 12-18
            - evening: 18-24
            - night: 0-6

    Args:
        df: Input dataframe
        time_column: Name of time column

    Returns:
        pd.DataFrame: Dataframe with time features

    Example:
        >>> df = create_time_features(df, 'Time')
        >>> df[['hour', 'minute', 'session']].head()
           hour  minute   session
        0    14      30  afternoon
        1    21      15    evening
        2     3      45     night
    """
```

##### `knn_impute(df, n_neighbors)`
```python
def knn_impute(
    df: pd.DataFrame,
    n_neighbors: int = 5
) -> pd.DataFrame:
    """
    Impute missing values using K-Nearest Neighbors.

    Args:
        df: Dataframe with missing values
        n_neighbors: Number of neighbors to use

    Returns:
        pd.DataFrame: Imputed dataframe

    Note:
        - Preserves data distribution
        - Works for both numerical and categorical features
        - Computationally expensive for large datasets
    """
```

##### `encode_features(df, target_column)`
```python
def encode_features(
    df: pd.DataFrame,
    target_column: str = 'Accident_severity'
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode categorical features and target variable.

    Encoding Strategy:
        - Categorical features: OrdinalEncoder
        - Target variable: LabelEncoder

    Args:
        df: Input dataframe
        target_column: Name of target column

    Returns:
        Tuple containing:
            - Encoded dataframe
            - Fitted LabelEncoder for target

    Example:
        >>> df_encoded, label_encoder = encode_features(df)
        >>> # Decode predictions
        >>> predictions = label_encoder.inverse_transform([0, 1, 2])
        >>> print(predictions)
        ['Minor', 'Severe', 'Fatal']
    """
```

##### `stratified_split(df, target_column, n_splits)`
```python
def stratified_split(
    df: pd.DataFrame,
    target_column: str = 'Accident_severity',
    n_splits: int = 5
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create stratified K-Fold splits.

    Args:
        df: Input dataframe
        target_column: Column to stratify on
        n_splits: Number of folds

    Yields:
        Tuple of (train_df, val_df) for each fold

    Example:
        >>> for fold, (train, val) in enumerate(stratified_split(df)):
        >>>     print(f"Fold {fold}: Train={len(train)}, Val={len(val)}")
        Fold 0: Train=9852, Val=2463
        Fold 1: Train=9852, Val=2463
        ...
    """
```

#### Complete Preprocessing Pipeline

```python
from src.preprocessing import (
    load_data,
    drop_unnecessary_columns,
    create_time_features,
    knn_impute,
    encode_features,
    stratified_split
)

# Load data
df = load_data("data/raw/RTA_Dataset.csv")

# Drop unnecessary columns
columns_to_drop = ['Time', 'Day_of_week', ...]
df = drop_unnecessary_columns(df, columns_to_drop)

# Create temporal features
df = create_time_features(df, time_column='Time')

# KNN imputation
df = knn_impute(df, n_neighbors=5)

# Encode features
df_encoded, label_encoder = encode_features(df)

# Create stratified splits
for fold, (train_df, val_df) in enumerate(stratified_split(df_encoded)):
    # Save splits
    train_df.to_csv(f"data/processed/fold_{fold}_train.csv", index=False)
    val_df.to_csv(f"data/processed/fold_{fold}_val.csv", index=False)
```

#### CLI Usage

```bash
# Standard usage
python src/preprocessing.py data/raw/ data/processed/

# With Makefile
make exe_preprocessing
```

---

### 3. Sampling Modules

#### Oversampling (`oversampler.py`)

##### `apply_smote(X, y, random_state)`
```python
def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique).

    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_resampled, y_resampled)

    Example:
        >>> from collections import Counter
        >>> print(f"Before: {Counter(y)}")
        Before: {0: 8621, 1: 3079, 2: 616}
        >>> X_res, y_res = apply_smote(X, y)
        >>> print(f"After: {Counter(y_res)}")
        After: {0: 8621, 1: 8621, 2: 8621}
    """
```

##### `apply_adasyn(X, y, random_state)`
```python
def apply_adasyn(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply ADASYN (Adaptive Synthetic Sampling).

    Focuses on generating samples for instances that are
    harder to classify (near decision boundary).

    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
```

#### Undersampling (`undersampler.py`)

##### `apply_tomek_links(X, y)`
```python
def apply_tomek_links(
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove Tomek links from dataset.

    Tomek link: Pair of samples from different classes
    that are each other's nearest neighbors.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        Tuple of (X_cleaned, y_cleaned)

    Note:
        - Cleans decision boundary
        - Removes noisy samples
        - Usually removes majority class samples
    """
```

#### Combined Sampling (`combined_sampling.py`)

##### `apply_smote_tomek(X, y, random_state)`
```python
def apply_smote_tomek(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE followed by Tomek Links cleaning.

    Pipeline:
        1. SMOTE: Generate synthetic samples
        2. TomekLinks: Remove noisy samples

    Args:
        X: Feature matrix
        y: Target vector
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)

    Example:
        >>> X_res, y_res = apply_smote_tomek(X, y)
        >>> # Balanced + cleaned data
    """
```

---

### 4. Model Training & Evaluation

#### Base Model Evaluation (`base_model_evaluation.py`)

##### `get_classification_models()`
```python
def get_classification_models() -> Dict[str, BaseEstimator]:
    """
    Return dictionary of classification models.

    Returns:
        Dict mapping model names to initialized estimators

    Models:
        - DecisionTree
        - RandomForest
        - GradientBoosting
        - AdaBoost
        - ExtraTrees
        - XGBoost
        - LightGBM
        - CatBoost
        - HistGradientBoosting

    Example:
        >>> models = get_classification_models()
        >>> for name, model in models.items():
        >>>     model.fit(X_train, y_train)
        >>>     score = f1_score(y_test, model.predict(X_test))
        >>>     print(f"{name}: {score:.3f}")
    """
```

##### `evaluate_model(model, X_train, y_train, X_test, y_test)`
```python
def evaluate_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Train and evaluate a single model.

    Args:
        model: Scikit-learn compatible estimator
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with metrics:
            - f1_weighted: Weighted F1-score
            - precision_macro: Macro precision
            - recall_macro: Macro recall
            - accuracy: Accuracy score
            - roc_auc_macro: Macro ROC-AUC

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        >>> print(f"F1-Score: {metrics['f1_weighted']:.3f}")
        F1-Score: 0.867
    """
```

#### Hyperparameter Tuning (`hp_tunnig_et.py`)

##### `bayesian_optimization(X_train, y_train, n_iter)`
```python
def bayesian_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 50
) -> Tuple[Dict, float]:
    """
    Perform Bayesian hyperparameter optimization.

    Args:
        X_train: Training features
        y_train: Training labels
        n_iter: Number of optimization iterations

    Returns:
        Tuple containing:
            - Best hyperparameters (dict)
            - Best F1-score (float)

    Search Space:
        n_estimators: [100, 500, 1000, 1500]
        max_depth: [10, 20, 30, 40, None]
        min_samples_split: [2, 5, 10, 15]
        min_samples_leaf: [1, 2, 4, 8]
        max_features: ['sqrt', 'log2', None]
        bootstrap: [True, False]

    Example:
        >>> best_params, best_score = bayesian_optimization(X, y, n_iter=50)
        >>> print(f"Best params: {best_params}")
        >>> print(f"Best F1: {best_score:.3f}")
        Best params: {'n_estimators': 1000, 'max_depth': 30, ...}
        Best F1: 0.893
    """
```

---

### 5. Explainability (XAI)

#### SHAP Analysis (`XAI_SHAP_v2.py`)

##### `compute_shap_values(model, X_background, X_explain)`
```python
def compute_shap_values(
    model: BaseEstimator,
    X_background: np.ndarray,
    X_explain: np.ndarray
) -> shap.Explanation:
    """
    Compute SHAP values for model predictions.

    Args:
        model: Trained tree-based model
        X_background: Background dataset for baseline
        X_explain: Instances to explain

    Returns:
        SHAP Explanation object

    Example:
        >>> shap_values = compute_shap_values(model, X_train, X_test)
        >>> # Visualize
        >>> shap.plots.waterfall(shap_values[0])
        >>> shap.plots.beeswarm(shap_values)
    """
```

##### `plot_shap_waterfall(shap_values, instance_idx, feature_names)`
```python
def plot_shap_waterfall(
    shap_values: shap.Explanation,
    instance_idx: int = 0,
    feature_names: List[str] = None,
    save_path: str = None
):
    """
    Create waterfall plot for single prediction.

    Args:
        shap_values: SHAP explanation object
        instance_idx: Index of instance to plot
        feature_names: List of feature names
        save_path: Path to save plot (optional)

    Example:
        >>> plot_shap_waterfall(
        ...     shap_values,
        ...     instance_idx=0,
        ...     feature_names=feature_names,
        ...     save_path="waterfall_plot.png"
        ... )
    """
```

##### `plot_shap_summary(shap_values, X, feature_names)`
```python
def plot_shap_summary(
    shap_values: shap.Explanation,
    X: np.ndarray,
    feature_names: List[str],
    plot_type: str = "bar"
):
    """
    Create SHAP summary plot.

    Args:
        shap_values: SHAP explanation object
        X: Feature matrix
        feature_names: List of feature names
        plot_type: Type of plot ('bar', 'beeswarm', 'violin')

    Example:
        >>> plot_shap_summary(shap_values, X_test, features, plot_type="beeswarm")
    """
```

---

## 🌐 Web Application

### Streamlit App (`streamlit_app.py`)

#### Main Components

##### `load_model()`
```python
@st.cache_resource
def load_model() -> BaseEstimator:
    """
    Load trained model (cached).

    Returns:
        Trained ExtraTreesClassifier

    Note:
        Uses Streamlit's @st.cache_resource for persistence
    """
```

##### `preprocess_input(input_data)`
```python
def preprocess_input(
    input_data: Dict[str, Any]
) -> np.ndarray:
    """
    Preprocess user input for prediction.

    Args:
        input_data: Dictionary of feature values

    Returns:
        Preprocessed feature array

    Pipeline:
        1. Validate input
        2. Create time features
        3. Encode categorical
        4. Scale numerical
        5. Select top features
    """
```

##### `predict_severity(model, features)`
```python
def predict_severity(
    model: BaseEstimator,
    features: np.ndarray
) -> Tuple[str, Dict[str, float]]:
    """
    Predict accident severity.

    Args:
        model: Trained model
        features: Preprocessed features

    Returns:
        Tuple containing:
            - Predicted class ('Minor', 'Severe', 'Fatal')
            - Confidence scores for each class

    Example:
        >>> prediction, probabilities = predict_severity(model, features)
        >>> print(f"Prediction: {prediction}")
        >>> print(f"Confidence: {probabilities[prediction]:.2%}")
        Prediction: Severe
        Confidence: 78.5%
    """
```

##### `display_shap_explanation(model, features)`
```python
def display_shap_explanation(
    model: BaseEstimator,
    features: np.ndarray
):
    """
    Display SHAP explanation in Streamlit.

    Args:
        model: Trained model
        features: Features to explain

    Displays:
        - Waterfall plot
        - Top contributing features
        - Feature importance bars
    """
```

#### Running the App

```bash
# Standard
streamlit run src/streamlit_app.py

# With options
streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

# Development mode (auto-reload)
streamlit run src/streamlit_app.py --server.runOnSave=true
```

---

## 💻 Code Examples

### Example 1: Complete Training Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import ADASYN

# 1. Load preprocessed data
train = pd.read_csv("data/processed/fold_0_train.csv")
test = pd.read_csv("data/processed/fold_0_val.csv")

# 2. Separate features and target
X_train = train.drop('Accident_severity', axis=1)
y_train = train['Accident_severity']
X_test = test.drop('Accident_severity', axis=1)
y_test = test['Accident_severity']

# 3. Apply ADASYN sampling
sampler = ADASYN(random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)

# 4. Train model
model = ExtraTreesClassifier(
    n_estimators=1000,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_res, y_train_res)

# 5. Evaluate
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"F1-Score: {f1:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Save model
import joblib
joblib.dump(model, 'models/best_model.pkl')
```

### Example 2: Feature Importance Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

# Load data
X_train = pd.read_csv("data/processed/fold_0_train.csv")
y_train = X_train.pop('Accident_severity')

# Train model
model = ExtraTreesClassifier(random_state=42)
model.fit(X_train, y_train)

# Get feature importances
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Display top 10
print(feature_importance.head(10))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance.head(10)['feature'],
    feature_importance.head(10)['importance']
)
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png')
```

### Example 3: SHAP Explanation

```python
import shap
import joblib
import numpy as np

# Load model and data
model = joblib.load('models/best_model.pkl')
X_test = pd.read_csv("data/processed/fold_0_val.csv")
y_test = X_test.pop('Accident_severity')

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Waterfall plot for first prediction
shap.plots.waterfall(shap_values[0])

# Summary plot (beeswarm)
shap.plots.beeswarm(shap_values)

# Feature importance
shap.plots.bar(shap_values)

# Save SHAP values
joblib.dump(shap_values, 'shap_values.joblib')
```

### Example 4: Batch Prediction

```python
import pandas as pd
import joblib
from typing import List, Dict

# Load model
model = joblib.load('models/best_model.pkl')

def batch_predict(
    model,
    data: pd.DataFrame
) -> List[Dict]:
    """
    Make predictions on batch of data.

    Args:
        model: Trained model
        data: DataFrame with features

    Returns:
        List of prediction dictionaries
    """
    # Predict
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    # Format results
    results = []
    classes = ['Minor', 'Severe', 'Fatal']

    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        results.append({
            'index': i,
            'prediction': classes[pred],
            'confidence': probs[pred],
            'probabilities': {
                classes[j]: prob
                for j, prob in enumerate(probs)
            }
        })

    return results

# Example usage
test_data = pd.read_csv("test_data.csv")
predictions = batch_predict(model, test_data)

# Display results
for result in predictions[:5]:  # First 5
    print(f"Sample {result['index']}:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print()
```

---

## 🖥️ CLI Reference

### Makefile Commands

```makefile
# Installation
make install                    # Install all dependencies

# Data Pipeline
make download_data              # Download dataset from Kaggle
make exe_preprocessing          # Preprocess raw data

# Sampling
make gen_oversampler            # Generate oversampled datasets
make gen_undersampler           # Generate undersampled datasets
make gen_combined_sampling      # Generate combined sampling datasets
make analyze_synthetic_data     # Analyze and rank sampling techniques

# Model Training
make base_model_evaluation      # Evaluate base models
make base_model_evaluation_FS   # Evaluate with feature selection
make feature_selection          # Extract feature importances
make feature_selection_eval     # Evaluate with selected features

# Optimization
make hyper_parametr_tuning      # Tune hyperparameters
make Eval_hyper_parametr_tuning # Evaluate tuned model

# Complete Pipeline
make setup                      # Run entire pipeline end-to-end
```

### Direct Python Execution

```bash
# Data download
python src/get_dataset.py

# Preprocessing
python src/preprocessing.py data/raw/ data/processed/

# Oversampling
python src/oversampler.py data/processed/ data/processed/sample_data/

# Undersampling
python src/undersampler.py data/processed/ data/processed/sample_data/

# Combined sampling
python src/combined_sampling.py data/processed/ data/processed/sample_data/

# Analyze sampling
python src/Analyzing_Synthetic_Data.py

# Base model evaluation
python src/base_model_evaluation.py

# Feature selection
python src/feature_selection.py

# Feature evaluation
python src/feature_selection_analysis.py

# Hyperparameter tuning
python src/hp_tunnig_et.py

# Final training
python src/train_and_evaluate_model.py

# SHAP analysis
python src/XAI_SHAP_v2.py

# Web app
streamlit run src/streamlit_app.py
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Kaggle API
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"

# Model configuration
export MODEL_PATH="models/best_model.pkl"
export DATA_PATH="data/processed/"

# Streamlit
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
```

### Configuration Files

#### `config.yaml` (example)

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  sample_dir: "data/processed/sample_data"

preprocessing:
  knn_neighbors: 5
  n_folds: 5
  random_state: 42

sampling:
  methods:
    - smote
    - adasyn
    - smote_tomek
  random_state: 42

models:
  base_models:
    - DecisionTree
    - RandomForest
    - ExtraTrees
    - XGBoost
    - LightGBM
    - CatBoost

  best_model: "ExtraTrees"

  hyperparameters:
    n_estimators: 1000
    max_depth: 30
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: "sqrt"
    bootstrap: true
    class_weight: "balanced"

evaluation:
  metrics:
    - f1_weighted
    - precision_macro
    - recall_macro
    - accuracy

  cv_folds: 5
```

---

## 🚨 Error Handling

### Common Errors

#### 1. Kaggle API Errors

```python
try:
    download_dataset("dataset_name")
except EnvironmentError:
    print("Error: Kaggle credentials not found")
    print("Solution: Set KAGGLE_USERNAME and KAGGLE_KEY")
except Exception as e:
    print(f"Download failed: {e}")
```

#### 2. Data Loading Errors

```python
try:
    df = pd.read_csv("data/raw/RTA_Dataset.csv")
except FileNotFoundError:
    print("Error: Dataset not found")
    print("Solution: Run 'make download_data' first")
except pd.errors.EmptyDataError:
    print("Error: Dataset is empty")
```

#### 3. Model Loading Errors

```python
import joblib

try:
    model = joblib.load("models/best_model.pkl")
except FileNotFoundError:
    print("Error: Model file not found")
    print("Solution: Train model using 'make Eval_hyper_parametr_tuning'")
except Exception as e:
    print(f"Model loading failed: {e}")
```

---

## ⚡ Performance Optimization

### Tips for Faster Execution

#### 1. Parallel Processing

```python
from sklearn.ensemble import ExtraTreesClassifier

# Use all CPU cores
model = ExtraTreesClassifier(n_jobs=-1)

# Or specify number of cores
model = ExtraTreesClassifier(n_jobs=4)
```

#### 2. Memory Optimization

```python
import pandas as pd

# Use categorical dtype for categorical features
df['feature'] = df['feature'].astype('category')

# Read only necessary columns
df = pd.read_csv("file.csv", usecols=['col1', 'col2'])

# Use chunking for large files
for chunk in pd.read_csv("large_file.csv", chunksize=10000):
    process(chunk)
```

#### 3. Caching with Streamlit

```python
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def preprocess_data(df):
    return process(df)
```

---

## 📖 Best Practices

### Code Quality

```python
# Good: Type hints
def predict(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    return model.predict(X)

# Good: Docstrings
def train_model(X, y):
    """
    Train classification model.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        Trained model
    """
    pass

# Good: Error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    handle_error()
```

### Model Serialization

```python
import joblib

# Save model
joblib.dump(model, 'model.pkl', compress=3)

# Load model
model = joblib.load('model.pkl')

# Save with metadata
metadata = {
    'model': model,
    'feature_names': feature_names,
    'hyperparameters': params,
    'metrics': metrics
}
joblib.dump(metadata, 'model_with_meta.pkl')
```

### Reproducibility

```python
import random
import numpy as np

# Set all random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # For other libraries
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
```

---

## 📞 Support

For technical support:
- 📧 Email: mianashfaque@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/zubairashfaque/Road-Traffic-Accident-Classification/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/zubairashfaque/Road-Traffic-Accident-Classification/discussions)

---

<div align="center">

**📚 Complete technical reference for the RTA Classification system**

Last Updated: March 2024

[⬆️ Back to Top](#-api-documentation---road-traffic-accident-classification)

</div>
