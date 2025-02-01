# House Price Prediction

This project demonstrates statistical modeling, data analysis, and machine learning using open-source tools on Ubuntu. It predicts house prices based on various features using a trained regression model.

## Overview
<img width="1223" alt="Screenshot 2025-02-01 at 2 22 32 am" src="https://github.com/user-attachments/assets/381ff81f-4c16-4b4d-84a3-dc4ff4ffc09a" />

### Explanation
- **Data Collection** → Uses a dataset (CSV) stored in `/data/`.
- **Data Preprocessing** → Cleans and prepares data (encoding, scaling).
- **Training the Model** → Trains a Linear Regression model, evaluates it, and saves it using `joblib`.
- **Model Deployment** → Loads the trained model for predictions.
- **User Interaction** → CLI-based input for house features, outputs predicted price.

## Features
- Data preprocessing and feature engineering
- Machine learning model training (Linear Regression)
- Model evaluation (MAE, MSE, R² Score)
- Predict house prices using a trained model
- Uses Scikit-learn, Pandas, NumPy, and Joblib

## Installation
### 1. Clone the repository:
```
git clone https://github.com/MenakaGodakanda/house_price_prediction.git
cd house_price_prediction
```
### 2. Set up a virtual environment:
```
python3 -m venv venv
source venv/bin/activate
```
![Screenshot 2025-02-01 012601](https://github.com/user-attachments/assets/e3f0d5b1-f624-4ae4-b1ee-9b894f7417f0)

### 3. Install dependencies:
We will use `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `jupyter`:
```
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
## Getting the Dataset
We will use the Kaggle House Prices Dataset. Download the dataset and place it inside `data/`.
```
mkdir data
cd data
wget https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv
```
![Screenshot 2025-02-01 012813](https://github.com/user-attachments/assets/674ab6b0-a2d6-4e78-b980-bec41b6fa442)
![Screenshot 2025-02-01 012802](https://github.com/user-attachments/assets/c50f287f-99b6-4dd0-863f-e48e488dd39b)

## Exploratory Data Analysis (EDA)
Create a Jupyter notebook inside notebooks/. Start Jupyter Notebook:
```
jupyter notebook
```
![Screenshot 2025-02-01 023927](https://github.com/user-attachments/assets/885ad912-5b88-47e2-a72d-eee534ff52e5)

### 1. Load the dataset:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = "../data/housing.csv"
df = pd.read_csv(data_path)

# Show first few rows
df.head()
```
![Screenshot 2025-02-01 013456](https://github.com/user-attachments/assets/1bcba95f-4f0d-49cb-bef2-6599e049ee25)

### 2. Basic Data Info:
```
df.info()
```
![Screenshot 2025-02-01 013550](https://github.com/user-attachments/assets/cda62f38-ddd9-436c-8971-7267f5b6f111)

### 3. Summary Statistics:
```
df.describe()
```
![Screenshot 2025-02-01 013605](https://github.com/user-attachments/assets/c58cd65a-0e1b-4ca8-8dac-19c2db70298e)

### 4. Data Visualization:
```
plt.figure(figsize=(8,6))
sns.histplot(df['median_house_value'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.show()
```
![Screenshot 2025-02-01 013628](https://github.com/user-attachments/assets/65f84790-62bf-4162-a226-447e8d991a14)
<br><br>
![Screenshot 2025-02-01 013640](https://github.com/user-attachments/assets/ad5c91f5-9353-4028-adc7-aa6dca124480)

### 5. Data Preprocessing:
#### Handling Missing Values
Check for missing values
```
print(df.isnull().sum())
```
![Screenshot 2025-02-01 013703](https://github.com/user-attachments/assets/97e7ae9e-0d18-4f73-b6be-49b6198fc05e)

Fill missing values with median
```
# Fill numeric columns with median
df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).fillna(df.median())

# Fill categorical columns with mode (most frequent value)
df[df.select_dtypes(include=['object']).columns] = df.select_dtypes(include=['object']).fillna(df.mode().iloc[0])
```
#### Encoding Categorical Data
```
df = pd.get_dummies(df, drop_first=True)
```

#### Feature Scaling
```
from sklearn.preprocessing import StandardScaler
import pandas as pd

# One-hot encode categorical variables (excluding target column)
df_encoded = pd.get_dummies(df.drop('median_house_value', axis=1), drop_first=True)

# Initialize the scaler
scaler = StandardScaler()

# Scale only the numeric features
scaled_features = scaler.fit_transform(df_encoded)

# Convert back to DataFrame with correct column names
df_scaled = pd.DataFrame(scaled_features, columns=df_encoded.columns)

# Add the target variable back
df_scaled['median_house_value'] = df['median_house_value']

# Display the first few rows
df_scaled.head()
```

### 6. Splitting Data & Building a Model:
#### Train-Test Split
```
from sklearn.model_selection import train_test_split

X = df_scaled.drop('median_house_value', axis=1)
y = df_scaled['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```
print(X_train.shape, X_test.shape)
```
![Screenshot 2025-02-01 014158](https://github.com/user-attachments/assets/7ad6a081-64d2-4cbe-a591-71c129c1f79a)

Training the Model
```
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Model Coefficients
print("Model Coefficients:", model.coef_)
```
![Screenshot 2025-02-01 013941](https://github.com/user-attachments/assets/a008e21c-6054-4baf-b2d7-7e98a129bb46)

### 7. Model Evaluation:
```
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
```

###  8. Saving the Model
```
import joblib

joblib.dump(model, "../models/house_price_model.pkl")
```


## Train the Model
Ensure the dataset is available in the data/ directory. Run the training script:
```
python3 src/train.py
```
This will preprocess the data, train a regression model, and save the trained model in `models/`.
![Screenshot 2025-02-01 015304](https://github.com/user-attachments/assets/03c9fddd-e8f0-4ab6-bc75-8a0c1aaa6e66)

## Making Predictions
Use the trained model to predict house prices:
```
python3 src/predict.py -118.32 34.21 25.0 4000.0 800.0 1500.0 750.0 5.0 "NEAR OCEAN"
```
Example Output:
![Screenshot 2025-02-01 015619](https://github.com/user-attachments/assets/8250619e-e985-425e-91a0-03581925af97)


## Troubleshooting
### FileNotFoundError
- Ensure you have trained the model before predicting: `python3 src/train.py`

### ValueError
- Make sure `predict.py` uses the same feature order as `train.py`.
- Re-run `train.py` and verify the feature count.

## Project Structure
```
house_price_prediction/
│── data/                    # Dataset storage
│── models/                  # Trained model storage
│── src/                     # Source code
│   │── train.py             # Model training script
│   │── predict.py           # Prediction script
│── README.md                # Project documentation
```

## License

This project is open-source under the MIT License.
