# Titanic-Survivors-analysis
# RMS Titanic - Passenger Manifest Analysis

A Streamlit application for exploring the Kaggle "Titanic - Machine Learning from Disaster" dataset. It performs data cleaning, visualizes survival patterns, evaluates classification models with honest cross-validation, and provides both single‑passenger prediction and batch submission generation.

## Key Features

- **Data cleaning and feature engineering**  
  Extracts titles from names, imputes missing Age by title group median, fills Embarked and Fare, creates FamilySize, IsAlone, and CabinKnown features.

- **Exploratory analysis**  
  Interactive charts showing survival rates by sex, class, age group, family size, title, fare quartile, and port of embarkation.

- **Model evaluation**  
  Logistic Regression and Random Forest are evaluated using stratified k‑fold cross‑validation (out‑of‑fold predictions). Displays accuracy, confusion matrix, ROC curve, precision/recall/F1, and feature importance or coefficients.

- **Single passenger prediction**  
  Enter a passenger’s details to obtain a survival probability. For Logistic Regression, shows per‑feature contributions; for Random Forest, shows global feature importances.

- **Batch prediction**  
  Generate a Kaggle‑format `submission.csv` from a test file using the selected model.

