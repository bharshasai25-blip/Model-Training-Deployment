#LightGBM Regressor model training script
import pandas as pd
import numpy as np
import lightgbm as lgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.target_transforms import AutoDifferences, LocalStandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# 1. DATA PREPARATION
# Assuming your columns are: 'Crime Type', 'Year', 'Month', 'Crime Count'
df = pd.read_csv('Model Training Datasets/Train_df_XGBoost_LightGBM_Models.csv')
df.sort_values(["TYPE", "YEAR"], inplace=True)
df.reset_index(drop=True, inplace=True)

# We combine 'Crime Type' as unique_id and 'Year'/'Month' into a ds column
df['ds'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
df = df.rename(columns={'TYPE': 'unique_id', 'Monthly_Crime_Count_Type_wise': 'y'})

# Ensure relevant columns are in the right format
df_train = df[['unique_id', 'ds', 'y', 'MONTH', 'is_summer', 'is_holiday_season', 'is_spring', 'quarter', 'month_sin', 'month_cos', 'month_sq', 'summer_peak', 'holiday_peak']].copy()
df_train = df_train.sort_values(['unique_id', 'ds'])


# 2. DEFINE THE MODEL & FEATURES
# We include Month as a dynamic feature and use recursive-friendly lags
fcst = MLForecast(
    models=[lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, random_state=42, verbosity=-1)],
    freq='MS',                # Monthly Start frequency
    lags=[1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 18, 24],          # Previous 2 months + same month last year
    lag_transforms={
        1: [RollingMean(window_size=3), RollingMean(window_size=6), RollingMean(window_size=12)]},
    date_features=[],  # Tells model to look at month-of-year seasonality
    target_transforms=[AutoDifferences(max_diffs=1)] # Handles the 12-month crime seasonality
)

# 3. PERFORM CROSS-VALIDATION (The "Evaluation" Step)
# This simulates 3 separate 24-month forecasts from your historical data
cv_results = fcst.cross_validation(
    df=df_train,
    h=24,
    n_windows=3,
    step_size=24,
    static_features=[] # Treats 'Month' as a dynamic variable
)

# Calculate Evaluation Metrics
def calculate_metrics(df_cv, model_name):
    actual = df_cv['y']
    pred = df_cv[model_name]
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)*100
    return mae, mape

mae, mape = calculate_metrics(cv_results, 'LGBMRegressor')
print(f"Cross-Validation Results -> MAE: {mae:.3f}, MAPE: {mape:.2f}%")

import numpy as np

def get_wmape(df_cv, model_col='LGBMRegressor'):
    # Sum of all absolute errors across all months and crime types
    total_abs_error = np.sum(np.abs(df_cv['y'] - df_cv[model_col]))

    # Sum of all actual crime counts
    total_actuals = np.sum(df_cv['y'])

    # Calculate weighted percentage
    wmape = (total_abs_error / total_actuals) * 100
    return wmape

# Usage
score = get_wmape(cv_results)
print(f"Global wMAPE: {score:.2f}%")


#4. Calculating MAE, MAPE per crime type

def per_crime_type_metrics(df, model_col='LGBMRegressor'):
    """
    Calculates MAE and MAPE per unique_id (crime type)
    using sklearn metric functions.
    """

    results = []

    grouped = df.groupby('unique_id')

    for uid, group in grouped:
        actual = group['y']
        pred = group[model_col]

        mae = mean_absolute_error(actual, pred)
        mape = mean_absolute_percentage_error(actual, pred) * 100

        results.append({
            'unique_id': uid,
            'MAE': mae,
            'MAPE_%': mape
        })

    result_df = pd.DataFrame(results)
    return result_df.sort_values('MAPE_%', ascending=False).reset_index(drop=True)

#Usage

metrics_per_type = per_crime_type_metrics(cv_results, model_col='LGBMRegressor')
print(metrics_per_type)

#Calculating WMAPE per crime type

def per_id_wmape(df_cv, model_col='LGBMRegressor'):
    # Group by unique_id and calculate (sum of errors / sum of actuals)
    summary = df_cv.groupby('unique_id').apply(
        lambda x: (np.sum(np.abs(x['y'] - x[model_col])) / np.sum(x['y'])) * 100
    ).reset_index()

    summary.columns = ['unique_id', 'wMAPE']
    return summary

# Usage
type_analysis = per_id_wmape(cv_results)
print(type_analysis.sort_values('wMAPE'))


# 5. FINAL FIT & FUTURE PREDICTION
# After verifying performance, train on ALL data and project 2 years ahead
fcst.fit(df_train[['unique_id', 'ds', 'y', 'MONTH', 'is_summer', 'is_holiday_season', 'is_spring', 'quarter', 'month_sin', 'month_cos', 'month_sq', 'summer_peak', 'holiday_peak']], static_features=[], max_horizon=24)

print("Static Features:", fcst.ts.static_features_)

# Create future dynamic features (X_df) for the next 24 months
last_date = df_train['ds'].max()
future_dates = pd.date_range(start=last_date + pd.offsets.MonthBegin(1), periods=24, freq='MS')
uids = df_train['unique_id'].unique()

X_df = pd.DataFrame({
    'unique_id': [i for i in uids for _ in range(24)],
    'ds': list(future_dates) * len(uids)
})
X_df['MONTH'] = X_df['ds'].dt.month
X_df['is_summer'] = X_df['MONTH'].isin([5, 6, 7, 8]).astype(int)
X_df['is_holiday_season'] = X_df['MONTH'].isin([9, 10, 11, 12]).astype(int)
X_df['is_spring'] = X_df['MONTH'].isin([1, 2, 3, 4]).astype(int)
X_df['quarter'] = ((X_df['MONTH'] - 1) // 3) + 1
X_df['month_sin'] = np.sin(2 * np.pi * (X_df['MONTH'] - 1)/ 12)
X_df['month_cos'] = np.cos(2 * np.pi * (X_df['MONTH'] - 1)/ 12)
X_df['month_sq'] = X_df['MONTH'] ** 2
X_df['summer_peak'] = X_df['is_summer'] * X_df['MONTH']
X_df['holiday_peak'] = X_df['is_holiday_season'] * X_df['MONTH']

# Final Forecast
#print("Static Features:", fcst.ts.static_features_)
required_features = [
    col for col in df_train.columns
    if col not in ['unique_id', 'ds', 'y']
]

missing = [c for c in required_features if c not in X_df.columns]

if missing:
    print("Missing required features:", missing)
else:
    print("All required features present.")


final_forecast1 = fcst.predict(h=24, X_df=X_df)
print("\nFinal 24-Month Forecast:")
print(final_forecast1.head())

# save the trained model for future use
#import joblib
#joblib.dump(fcst, 'trained_LightGBM_forecast_model1.pkl')

#save the finalforecast

final_forecast1.to_csv("Full_forecast_LightGBM.csv", index=False)
print("Predictions saved successfully.")