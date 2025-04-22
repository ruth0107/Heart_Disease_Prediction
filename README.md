# Heart Disease Prediction App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

A Streamlit web application for predicting heart disease risk based on patient health metrics using a Random Forest Classifier.

## Features

- Upload and process CSV files containing patient health data
- Train a machine learning model (Random Forest Classifier) on the uploaded data
- Display model performance metrics including test accuracy
- Visualize model performance with a confusion matrix
- Interactive form for making predictions on new patient data

## Requirements

- Python 3.7+
- Streamlit
- scikit-learn
- pandas
- matplotlib

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. In the web interface:
   - Upload a CSV file containing heart disease data
   - The app will automatically train the model and show performance metrics
   - Use the prediction form to make predictions on new patient data

## Data Format

The application expects a CSV file with the following columns (adjustable in code):

- `age`: Patient age in years
- `sex`: Patient gender ('male' or 'female')
- `resting bp s`: Resting blood pressure in mmHg
- `cholesterol`: Cholesterol level in mg/dL
- `fasting blood sugar`: Fasting blood sugar > 120 mg/dL ('true' or 'false')
- `max heart rate`: Maximum heart rate achieved
- `exercise angina`: Exercise induced angina ('yes' or 'no')
- `target`: Target variable (1 = heart disease, 0 = no heart disease)

## Model Details

The application uses:
- Random Forest Classifier
- Data preprocessing pipeline including:
  - Standard scaling for numerical features
  - One-hot encoding for categorical features
  - Mean imputation for missing numerical values
  - Mode imputation for missing categorical values

## Screenshots

![App Screenshot](app-screenshot.png)

