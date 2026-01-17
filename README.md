# Quantified ADHD Diagnostic Support System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust Machine Learning application developed to assist in the quantitative assessment of ADHD. This system utilizes a calibrated Logistic Regression model, rigorously trained on clinical Continuous Performance Test (CPT-II) data, to provide objective diagnostic probabilities based on 75 identified biomarkers.

## System Capabilities

*   **High-Fidelity Modeling**: Implements a specialized Logistic Regression classifier validated with an ROC-AUC of ~0.98.
*   **Instant Analysis**: Processes standard raw patient CSV data to generate immediate diagnostic reports.
*   **Clinical Visualization**: Presents confidence intervals and probability metrics via a clean, high-contrast interface designed for clinical environments.
*   **Automated Validation**: Cross-references predictions against ground truth labels (when available) for continuous performance verification.

## Scientific Methodology

This tool supports clinical decision-making by analyzing objective performance metrics alongside demographic data.

### 1. Input Data: Conners' CPT-II
The core data source is the **Conners' Continuous Performance Test II (CPT-II)**, a standardized computerized assessment measuring:
-   **Sustained Attention**: The ability to maintain focus over extended periods.
-   **Impulsivity**: The tendency to respond incorrectly to non-target stimuli.

### 2. Feature Optimization (Random Forest)
From an initial high-dimensional feature space, a **Random Forest** algorithm was employed to compute feature importance scores. This process isolated the **top 75 predictive biomarkers**, filtering noise to retain only the most diagnostically relevant signals:
-   **Performance Metrics**: Omission errors, Commission errors, and Reaction Time (Hit RT) variability.
-   **Signal Processing**: Fourier Transform coefficients and Entropy measures extracted from reaction time series.
-   **Demographics**: Age-normed standardized scores.

### 3. Predictive Model (Logistic Regression)
The final determination is made by a **Logistic Regression** classifier. This model architecture was selected for its interpretability and calibrated probability outputs.
-   **High Probability (>50%)**: Indicates patterns consistent with ADHD pathology.
-   **Low Probability (<50%)**: Indicates patterns consistent with neurotypical attention control.

## Project Structure

```
├── data/               # Data repositories and CSV assets
│   ├── patient_files/  # Individual patient records for testing
│   └── *.csv           # Source datasets and processed features
├── src/                # Core application logic
│   ├── pred.py         # Machine Learning pipeline and inference engine
│   ├── prepare_data.py # Data preprocessing and feature engineering
│   └── split_patients.py # Utility for dataset segmentation
├── streamlit_app.py    # Main user interface entry point
└── requirements.txt    # Project dependencies
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/in-ritik/ADHD-Detection.git
    cd ADHD-Detection
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## usage

1.  **Initialize the Application:**
    ```bash
    streamlit run streamlit_app.py
    ```

2.  **Perform Analysis:**
    *   The interface will launch automatically in your default browser (http://localhost:8501).
    *   The model will pre-load necessary weights and configurations.
    *   Upload a patient CSV record from `data/patient_files/` to generate a comprehensive diagnostic report.

## Model Training

The application includes an automated retraining pipeline. On startup, it processes the source datasets (`features.csv`, `patient_info.csv`) to ensure the diagnostic model reflects the most current logic defined in `pred.py`.


