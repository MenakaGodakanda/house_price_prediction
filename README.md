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

## Installation

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
