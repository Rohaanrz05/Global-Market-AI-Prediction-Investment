# Global Market AI Prediction Investment

A professional Streamlit Machine Learning web application that analyzes global market data and predicts AI-based investment decisions using a Random Forest Classifier.

## Project Overview

This project uses a cleaned Global Market AI dataset to predict investment decisions such as **Invest**, **Avoid**, and **Monitor**. The application is built with Streamlit and includes a modern dashboard, dataset preview, analytics charts, ML model training, model evaluation, and live prediction.

## Features

- Professional Streamlit frontend
- Multi-tab dashboard design
- Dataset preview and summary
- Interactive charts using Plotly
- Machine Learning prediction tab
- Random Forest Classifier model
- Accuracy score and classification report
- Confusion matrix visualization
- Feature importance analysis
- Live investment prediction form

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## Machine Learning Model

The model used in this project is:

```text
Random Forest Classifier
```

The model predicts the target column:

```text
AI_Investment_Decision
```

Prediction classes:

```text
Invest
Avoid
Monitor
```

## Dataset

The dataset file used in this project is:

```text
clean_global_market_ai_dataset_small.csv
```

If you are using the full dataset, the file name can be:

```text
clean_global_market_ai_dataset_for_ml.csv
```

Important dataset features include:

```text
Market_Type
Asset_Type
Price
Trading_Volume
Market_Cap
Volatility_Index
Liquidity_Score
News_Sentiment
Social_Sentiment
Risk_Level
Expected_Return
Technical_Signal
Market_Trend
Inflation_Rate
Interest_Rate
Currency_Strength_Index
AI_Investment_Decision
```

## Project Structure

```text
Global-Market-AI-Prediction-Investment/
│
├── app.py
├── requirements.txt
├── README.md
└── clean_global_market_ai_dataset_small.csv
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

## How to Run

Run this command in the project folder:

```bash
streamlit run app.py
```

After running the command, the application will open in your browser.

## Application Tabs

### 1. Overview

Shows the project introduction, key features, and investment decision distribution.

### 2. Dataset

Displays the cleaned dataset, rows, columns, missing values, and column information.

### 3. Analytics

Shows interactive visualizations related to investment decisions, risk level, expected return, and sentiment scores.

### 4. ML Prediction

Allows the user to enter market values and get a live prediction from the trained Machine Learning model.

### 5. Model Details

Displays model accuracy, classification report, confusion matrix, and feature importance.

## Model Evaluation

The model is evaluated using:

```text
Accuracy Score
Classification Report
Confusion Matrix
Feature Importance
```

## Prediction Output

The model predicts one of the following decisions:

### Invest

The market condition looks favorable for investment.

### Avoid

The market condition looks risky, so investment should be avoided.

### Monitor

The market condition should be observed before making a final decision.

## Requirements

The `requirements.txt` file should contain:

```text
streamlit
pandas
numpy
scikit-learn
plotly
```

## Conclusion

This project demonstrates how Machine Learning can be applied to financial market analysis. The Streamlit frontend provides a professional and interactive interface for exploring data, viewing analytics, evaluating the model, and making live investment predictions.

## Author

Developed by **Hamza Afzal**
