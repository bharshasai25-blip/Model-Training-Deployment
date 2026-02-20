FBI Crime Record Analysis and Forecasting Project
Introduction

This project focuses on analyzing historical FBI crime data to identify trends, patterns, and forecast future crime counts using advanced machine learning and time series forecasting models. The project implements LightGBM, XGBoost, and SARIMAX models to predict crime counts across different crime types, months, and neighbourhoods. The best-performing models are deployed using Streamlit to enable real-time interactive forecasting.

The objective of this project is to support data-driven decision-making for crime prevention, resource allocation, and public safety planning.

Project Objectives

Analyze historical crime data to identify trends and patterns

Perform Exploratory Data Analysis (EDA) and visualization

Forecast future crime counts using machine learning and time series models

Compare model performance and accuracy

Deploy the best-performing model using Streamlit for real-time prediction

Dataset Description

The dataset contains historical FBI crime records with the following key features:

Year

Month

Crime Type

Neighbourhood

Monthly Crime Count

Yearly Crime Count

The dataset captures temporal and geographical crime trends across multiple years.

Exploratory Data Analysis (EDA)

EDA was performed to understand crime patterns and relationships between variables.

Key analysis performed:

Crime trend analysis over time

Crime distribution by crime type

Crime distribution by neighbourhood

Seasonal trend analysis

Crime density and clustering analysis

3D scatter plot visualization

Visualizations

The following visualizations were created:

Crime trend line plots (Year-wise and Month-wise)

Crime count distribution plots

Crime type comparison charts

Neighbourhood crime comparison charts

Heatmaps and correlation analysis

3D scatter plots for multi-dimensional pattern analysis

Forecasted vs actual crime trend comparison plots

These visualizations helped identify trends, seasonal patterns, and high-crime areas.

Models Implemented
1. LightGBM Regressor

LightGBM is a gradient boosting framework that uses tree-based learning algorithms.

Features:

High training speed

Efficient memory usage

Excellent performance on structured data

Captures complex patterns effectively

Used for:

Monthly crime count forecasting by crime type

2. XGBoost Regressor

XGBoost is an optimized gradient boosting algorithm known for its accuracy and robustness.

Features:

High prediction accuracy

Handles nonlinear relationships effectively

Strong generalization capability

Used for:

Monthly crime count forecasting by crime type

3. SARIMAX Model

SARIMAX is a statistical time series forecasting model.

Features:

Captures trend and seasonality

Strong interpretability

Suitable for time series forecasting

Used for:

Yearly crime count forecasting by neighbourhood

Model Performance Summary
Model	Forecast Type	Performance
LightGBM	Monthly Crime Forecast	Best Performance
XGBoost	Monthly Crime Forecast	Very Good Performance
SARIMAX	Yearly Crime Forecast	Good Performance

LightGBM provided the most accurate and stable predictions for monthly crime forecasting.

SARIMAX performed well for yearly forecasting but is less flexible than machine learning models.

Model Deployment

The best-performing model was deployed using Streamlit.

Deployment features:

Interactive user interface

Real-time crime prediction

User input-based forecasting

Easy-to-use web application

Tools used:

Streamlit

Python

VS Code

Technologies Used

Programming Language:

Python

Libraries:

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

LightGBM

XGBoost

Statsmodels

Streamlit

Tools:

Jupyter Notebook

VS Code

GitHub

Project Structure
Crime-Forecasting-Project/
│
├── FBI Crime Record Analysis Project.ipynb
├── dataset.csv
├── streamlit_app.py
├── models/
├── visualizations/
├── README.md
Key Insights

Crime patterns show strong seasonal and yearly trends

Certain crime types consistently have higher crime counts

Machine learning models outperform traditional time series models in monthly forecasting

LightGBM provided the best overall performance

Crime forecasting can help in proactive crime prevention

Future Improvements

Use larger and more recent datasets

Add deep learning models such as LSTM

Improve model hyperparameter tuning

Deploy using cloud platforms (AWS, Azure, GCP)

Build full production-ready web application

Conclusion

This project successfully analyzed and forecasted crime trends using machine learning and time series models. LightGBM and XGBoost demonstrated strong performance in forecasting monthly crime counts, while SARIMAX provided reliable yearly forecasts. The deployment of the model using Streamlit enables real-time interactive crime prediction, making the solution practical and useful for crime analysis and decision-making.
