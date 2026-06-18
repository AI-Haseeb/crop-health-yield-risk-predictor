
# Farmer Report: Yield Risk Prediction

## 1. Problem

Farmers need help to understand whether a recommended crop has low, medium, or high yield risk. My task was to build Model 2, which predicts yield risk for the crop recommended by Model 1.

## 2. Dataset

I used the Crop Yield Prediction Dataset. The dataset contains crop name, country/area, year, rainfall, pesticide use, average temperature, and yield.

The dataset did not already have a yield_risk column, so I created it using historical yield performance and weather volatility.

## 3. Model

I built an Artificial Neural Network, ANN, for yield risk classification.

The model predicts three classes:

- LOW risk
- MEDIUM risk
- HIGH risk

The model uses these inputs:

- Recommended crop
- Area
- Year
- Rainfall
- Average temperature
- Pesticide use
- Weather volatility features

## 4. Feature Engineering

I created weather volatility features to help the model understand risk:

- rainfall_volatility
- temp_volatility
- pesticide_volatility
- weather_volatility_score
- pesticides_log
- year_index

These features help the model compare current weather conditions with normal crop conditions.

## 5. Training

I used an ANN with ReLU activation in hidden layers and Softmax in the output layer.

Dropout was used to reduce overfitting. Early stopping was used to stop training when validation performance stopped improving.

Class weights were used to handle possible class imbalance.

## 6. Results

The model predicts yield risk as LOW, MEDIUM, or HIGH.

Evaluation includes:

- Accuracy
- F1-score per class
- Confusion matrix
- Training and validation graphs

## 7. Farmer Recommendation Logic

The final decision logic is:

- LOW risk means GO
- MEDIUM risk means CAUTION
- HIGH risk means HOLD

Example:

Recommended crop: Maize  
Yield risk: LOW  
Decision: GO  

This means the farmer can proceed because the predicted risk is low.

## 8. Limitations

The yield_risk label was created from historical data. It was not directly provided by farmers.

The model is for decision support only. It should not be treated as a final farming guarantee.

The final group pipeline still needs Model 1 to recommend the crop before Model 2 predicts risk.
