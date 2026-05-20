# Student Score Predictor

A machine learning web app that trains a regression model on a student dataset and predicts a selected score/target column.

## Live Demo

[Click here to view the deployed app](https://student-score-predictor-spys6shfzupqxkqducitzs.streamlit.app/)


## Features

- Upload any student CSV dataset
- Choose the target score column
- Select input features
- Train Linear Regression or Random Forest models
- View MAE, RMSE, and R² Score
- Predict a score using interactive inputs
- Download the trained model

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit

## How to Run Locally

```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

You can either:

1. Upload a CSV from the app sidebar, or
2. Place your dataset at:

```text
data/star98.csv
```

## Suggested GitHub Description

A Streamlit machine learning app that predicts student scores using regression models, with dataset upload, model evaluation, and interactive prediction.
