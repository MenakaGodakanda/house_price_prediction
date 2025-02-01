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
cd notebooks
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
![Screenshot 2025-02-01 153238](https://github.com/user-attachments/assets/80425664-5755-43e3-8668-72484de833c2)

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
![Screenshot 2025-02-01 153658](https://github.com/user-attachments/assets/fbb1be13-9e70-4ed0-bee8-fedc5c3b76b6)

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
![Screenshot 2025-02-01 153750](https://github.com/user-attachments/assets/85d86a49-413c-4591-9377-9581eea16bd1)

This means 16,512 data points for training and 4,128 for testing.

#### Training the Model
```
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# Model Coefficients
print("Model Coefficients:", model.coef_)
```
![Screenshot 2025-02-01 153811](https://github.com/user-attachments/assets/e7663db8-3889-4341-9865-0c5b1f2c8530)

These are the weights the model assigns to different features.

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
![Screenshot 2025-02-01 153836](https://github.com/user-attachments/assets/f7299dd1-81ed-424b-9083-e3ed7c2b73a7)

- **Mean Absolute Error (MAE)**: On average, the predicted house prices deviate by $50,670.74 from the actual prices.
- **Mean Squared Error (MSE)**: The large number shows the squared error, which penalizes larger mistakes more than smaller ones. Since house prices are in the hundreds of thousands, squared errors become very large.
- **Coefficient of Determination (R²)**: 62.54% of the variability in house prices is explained by the model.

**Is This Model Good?**
- Not bad, but not great
- 62.54% R² suggests a moderate model.
- The high MAE ($50K error) may need improvements, like:
  - Feature engineering (adding better predictors)
  - Non-linear models (e.g., Random Forest, XGBoost)
  - Hyperparameter tuning

###  8. Saving the Model
```
import joblib

joblib.dump(model, "../models/house_price_model.pkl")
```
![Screenshot 2025-02-01 153853](https://github.com/user-attachments/assets/6223fefd-f735-462e-b735-6363113dc0a8)

## Making Predictions
Use the trained model to predict house prices:
- Change the input data for prediction in `predict.py`
- Run `predict.py`:
```
python3 src/predict.py
```
Example Output:<br>
![Screenshot 2025-02-01 162214](https://github.com/user-attachments/assets/db91a892-77b1-4d24-bef8-2d6556324f5b)

## Project Structure
```
house_price_prediction/
│── data/                                # Dataset storage
│   │── housing.csv                      # Dataset
│── models/                              # Trained model storage
│   │── house_price_model.pkl            # Prediction model
│── src/                                 # Source code
│   │── predict.py                       # Prediction script
│── notebooks                            # Jupiter notebooks
```

## License

This project is open-source under the MIT License.
