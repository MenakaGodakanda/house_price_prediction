import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the model
model_path = "./models/house_price_model.pkl"
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load the dataset
data_path = "./data/housing.csv"
df = pd.read_csv(data_path)

# Define features and target
features = df.drop(columns=['median_house_value'])
target = df['median_house_value']

# Preprocessing for numerical data
numerical_features = features.select_dtypes(include=['float64']).columns
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_features = features.select_dtypes(include=['object']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit the preprocessor on the training data
preprocessor.fit(features)

# Example input data for prediction
example_input = {
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41.0],
    'total_rooms': [880.0],
    'total_bedrooms': [129.0],
    'population': [322.0],
    'households': [126.0],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
}

# Convert example input to DataFrame
example_df = pd.DataFrame(example_input)

# Preprocess the example input
processed_input = preprocessor.transform(example_df)

# Predict the house price
predicted_price = model.predict(processed_input)

# Display the predicted price
print(f"Predicted House Price: {predicted_price[0]:.2f}")
