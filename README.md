# TQIP Local Features Format Guide

## Overview
This document explains how to prepare input data formats compatible with `tqip_nei6_local_2018.py` for use by other collaborators.

## Required Field Descriptions

### 1. Basic Identification Fields
- **inc_key**: Patient unique identifier (string format, e.g., "TQIP2018001")

### 2. Basic Demographic Features
- **age**: Age (numeric, unit: years)
- **sex**: Gender (numeric encoding: 1=male, 2=female)

### 3. Hospital Vital Signs
- **total_gcs**: Glasgow Coma Scale total score (numeric, range 3-15)
- **sbp**: Systolic blood pressure (numeric, unit: mmHg)

### 4. Time-related Features
- **arrival_hour**: Arrival time (numeric, range 0-23, 24-hour format)
- **evening_arrival**: Evening arrival (binary variable: 0=no, 1=yes)
  - Definition: arrival_hour >= 18 or arrival_hour <= 6

### 5. Trauma Mechanism Features
- **firearm_injury**: Firearm injury (binary variable: 0=no, 1=yes)
- **fall_injury**: Fall injury (binary variable: 0=no, 1=yes)
- **unintentional_injury**: Unintentional injury (binary variable: 0=no, 1=yes)
- **central_gunshot_wound**: Central gunshot wound (binary variable: 0=no, 1=yes)

### 6. Target Variable
- **NEI6_positive**: NEI-6 positive (binary variable: 0=no, 1=yes)

## Data Format Requirements

### Numeric Fields
- All numeric fields must be in numeric format, cannot contain text
- Missing values represented as NaN or empty values
- Age range recommended: 0-120 years
- GCS total score range: 3-15
- Systolic blood pressure range: 50-300 mmHg
- Arrival time: 0-23

### Binary Variables
- All binary variables use 0 and 1 encoding
- 0 = No/None
- 1 = Yes/Have

### String Fields
- inc_key: Unique identifier, recommended format "TQIP2018XXX"

## Sample Data

Please refer to the `sample_tqip_local_features.csv` file, which contains 50 sample records.

## Data Quality Requirements

### Completeness
- The model automatically handles missing values, but it is recommended to provide complete data for optimal prediction performance
- If a feature is completely missing, that feature will be excluded from the model

### Data Range Validation
- Age: 0-120 years
- GCS: 3-15
- Systolic blood pressure: 50-300 mmHg
- Arrival time: 0-23

## Usage Instructions

### 1. Data Preparation
```python
import pandas as pd

# Read your data
df = pd.read_csv('your_data.csv')

# Ensure column names are correct
required_columns = [
    'inc_key', 'age', 'sex', 'total_gcs', 'sbp', 'arrival_hour', 
    'evening_arrival', 'firearm_injury', 'fall_injury', 
    'unintentional_injury', 'central_gunshot_wound', 'NEI6_positive'
]

# Check if required columns exist
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing required columns: {missing_columns}")
```

### 2. Execute Prediction
```python
from tqip_nei6_local_2018 import TQIPNEI6LocalPredictor2018

# Create predictor
predictor = TQIPNEI6LocalPredictor2018()

# Load and process data
df = predictor.load_data_2018()  # or use your data
df = predictor.create_local_nei6_labels_2018(df)
df = predictor.create_local_features_2018(df)

# Prepare features
X, y = predictor.prepare_features_local_2018(df)

# Train model
best_result = predictor.train_models_2018(X, y)
```

## Important Notes

1. **Data Encoding**: The gender field will be automatically label-encoded, please ensure to use 1 and 2 to represent male and female
2. **Missing Value Handling**: The model uses a conservative strategy, only keeping completely complete records
3. **Feature Selection**: If certain features are completely missing, the model will automatically exclude these features
4. **Data Balance**: It is recommended to ensure a reasonable distribution of NEI6_positive to avoid extreme imbalance

---
*Last updated: January 2025*
