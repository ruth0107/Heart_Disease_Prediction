import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

@st.cache_data
def load_data(uploaded_file):
    """Loads the dataset from the uploaded file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading the CSV file: {e}")
        return None

def train_model(df):
    """Trains the machine learning model."""
    if df is None:
        return None

    # Inspect columns to verify names
    st.write("Available columns in dataset:")
    st.write(df.columns)

    # Define feature columns and target variable based on available columns
    # Adjust these names based on the actual column names
    feature_columns = ['age', 'sex', 'resting bp s', 'cholesterol', 'fasting blood sugar',
                       'max heart rate', 'exercise angina']  # Adjusted column names
    target_column = 'target'  # The target variable remains the same

    # Ensure the columns exist in the dataset
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
        return None
    else:
        # Separate features (X) and target (y)
        X = df[feature_columns]
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define preprocessing steps for numerical and categorical columns
        numeric_features = ['age', 'resting bp s', 'cholesterol', 'max heart rate']
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_features = ['sex', 'fasting blood sugar', 'exercise angina']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Create a pipeline with preprocessing and the classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        # Fit the pipeline on the training data
        pipeline.fit(X_train, y_train)
        st.success("Model training complete.")

        # Evaluate the pipeline on the test data
        test_score = pipeline.score(X_test, y_test)
        st.write(f"Test Accuracy: {test_score:.2f}")

        # Generate predictions
        y_pred = pipeline.predict(X_test)

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.named_steps['classifier'].classes_)
        disp.plot(cmap="Blues")
        st.pyplot(plt.gcf())  # Use st.pyplot to display the plot

        return pipeline

def predict_patient_health(pipeline):
    """Predicts whether a patient has heart disease based on user input."""
    if pipeline is None:
        st.warning("Model not trained yet. Please train the model first.")
        return

    st.header("Patient Health Prediction")

    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    sex = st.selectbox("Sex", options=['male', 'female'])
    resting_bp = st.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=140)
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=600, value=289)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=['false', 'true'])
    max_heart_rate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=172)
    exercise_angina = st.selectbox("Exercise Angina", options=['no', 'yes'])

    new_patient = {
        'age': age,
        'sex': sex,
        'resting bp s': resting_bp,
        'cholesterol': cholesterol,
        'fasting blood sugar': fasting_blood_sugar,
        'max heart rate': max_heart_rate,
        'exercise angina': exercise_angina
    }

    new_patient_df = pd.DataFrame([new_patient])
    prediction = pipeline.predict(new_patient_df)  # Extract the first element from the array

    if st.button("Predict"):
        if prediction == 1:
            st.write("The patient has heart disease.")
        else:
            st.write("The patient does not have heart disease.")

def main():
    """Main function to run the Streamlit app."""
    st.title("Heart Disease Prediction App")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            pipeline = train_model(df)
            predict_patient_health(pipeline)
    else:
        st.info("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
