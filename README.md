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
## Exploratory Data Analysis (EDA)
Create a Jupyter notebook inside notebooks/. Start Jupyter Notebook:
```
jupyter notebook
```

### 1. Load the dataset:

### 2. Basic Data Info:
```
df.info()
```

### 3. Summary Statistics:
```
df.describe()
```

### 4. Data Visualization:
```
plt.figure(figsize=(8,6))
sns.histplot(df['median_house_value'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.show()
```

### 5. Data Preprocessing:
Handling Missing Values
```
print(df.isnull().sum())
```

### 6. Model Training:
Train-Test Split
```
print(X_train.shape, X_test.shape)
```

Training the Model
```
print("Model Coefficients:", model.coef_)
```

### 7. Model Evaluation:
```
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
```

### 8. Making Predictions:
```
sample_data = [[-118.32, 34.21, 25.0, 4000.0, 800.0, 1500.0, 750.0, 5.0]]
prediction = model.predict(sample_data)
print(f"Predicted House Price: {prediction[0]}")
```

## Train the Model
Ensure the dataset is available in the data/ directory. Run the training script:
```
python3 src/train.py
```
This will preprocess the data, train a regression model, and save the trained model in `models/`.

## Making Predictions
Use the trained model to predict house prices:
```
python3 src/predict.py -118.32 34.21 25.0 4000.0 800.0 1500.0 750.0 5.0 "NEAR OCEAN"
```
Example Output:

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
