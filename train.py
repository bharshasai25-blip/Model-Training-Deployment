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
df = pd.read_csv('D:\Module 6 Project\Model Training Datasets\Train_df_XGBoost_LightGBM_Models.csv')
df.sort_values(["TYPE", "YEAR"], inplace=True)
df.reset_index(drop=True, inplace=True)

# We combine 'Crime Type' as unique_id and 'Year'/'Month' into a ds column
df['ds'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
df = df.rename(columns={'TYPE': 'unique_id', 'Monthly_Crime_Count_Type_wise': 'y'})

# Ensure relevant columns are in the right format
df_train = df[['unique_id', 'ds', 'y', 'MONTH', 'is_summer', 'is_holiday_season', 'is_spring', 'quarter', 'month_sin', 'month_cos', 'month_sq', 'summer_peak', 'holiday_peak']].copy()

# 2. DEFINE THE MODEL & FEATURES
# We include Month as a dynamic feature and use recursive-friendly lags
fcst = MLForecast(
    models=[lgb.LGBMRegressor(n_estimators=1200, learning_rate=0.02, random_state=42, verbosity=-1)],
    freq='MS',                # Monthly Start frequency
    lags=[1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 18, 24],          # Previous 2 months + same month last year
    lag_transforms={
        1: [RollingMean(window_size=3), RollingStd(window_size=3), RollingMean(window_size=6), RollingStd(window_size=6), RollingMean(window_size=12), RollingStd(window_size=12), RollingMean(window_size=24)]
    },
    date_features=['month'],  # Tells model to look at month-of-year seasonality
    target_transforms=[AutoDifferences(max_diffs=2)] # Handles the 12-month crime seasonality
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
fcst.fit(df_train, static_features=[], max_horizon=24)

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
final_forecast1 = fcst.predict(h=24, X_df=X_df)
print("\nFinal 24-Month Forecast:")
print(final_forecast1.head())

# save the trained model for future use
import joblib
joblib.dump(fcst, 'trained_LightGBM_forecast_model.pkl')



#XGBoost Regressor model training script
import pandas as pd
import numpy as np
import xgboost as xgb
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from mlforecast.target_transforms import AutoDifferences, LocalStandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# 1. DATA PREPARATION
# Assuming your columns are: 'Crime Type', 'Year', 'Month', 'Crime Count'
df = pd.read_csv('D:\Module 6 Project\Model Training Datasets\Train_df_XGBoost_LightGBM_Models.csv')
df.sort_values(["TYPE", "YEAR"], inplace=True)
df.reset_index(drop=True, inplace=True)

# We combine 'Crime Type' as unique_id and 'Year'/'Month' into a ds column
df['ds'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
df = df.rename(columns={'TYPE': 'unique_id', 'Monthly_Crime_Count_Type_wise': 'y'})

# Ensure relevant columns are in the right format
df_train = df[['unique_id', 'ds', 'y', 'MONTH', 'is_summer', 'is_holiday_season', 'is_spring', 'quarter', 'month_sin', 'month_cos', 'month_sq', 'summer_peak', 'holiday_peak']].copy()

# 2. DEFINE THE MODEL & FEATURES
# We include Month as a dynamic feature and use recursive-friendly lags
fcst = MLForecast(
    models=[xgb.XGBRegressor(enable_categorical=True, tree_method="hist", n_estimators=1200, learning_rate=0.02, random_state=42, verbosity=0)],
    freq='MS',                # Monthly Start frequency
    lags=[1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 18, 24],          # Previous 2 months + same month last year
    lag_transforms={
        1: [RollingMean(window_size=3), RollingStd(window_size=3), RollingMean(window_size=6), RollingStd(window_size=6), RollingMean(window_size=12), RollingStd(window_size=12), RollingMean(window_size=24)]
    },
    date_features=['month'],  # Tells model to look at month-of-year seasonality
    target_transforms=[AutoDifferences(max_diffs=2)] # Handles the 12-month crime seasonality
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

mae, mape = calculate_metrics(cv_results, 'XGBRegressor')
print(f"Cross-Validation Results -> MAE: {mae:.3f}, MAPE: {mape:.2f}%")

import numpy as np

def get_wmape(df_cv, model_col='XGBRegressor'):
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

def per_crime_type_metrics(df, model_col='XGBRegressor'):
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

metrics_per_type = per_crime_type_metrics(cv_results, model_col='XGBRegressor')
print(metrics_per_type)

#Calculating WMAPE per crime type

def per_id_wmape(df_cv, model_col='XGBRegressor'):
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
fcst.fit(df_train, static_features=[], max_horizon=24)

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
final_forecast4 = fcst.predict(h=24, X_df=X_df)
print("\nFinal 24-Month Forecast:")
print(final_forecast4.head())

# save the trained model for future use
import joblib
joblib.dump(fcst, 'trained_XGBoost_forecast_model.pkl')




#SARIMA Regressor model training script
# ==========================================================
# 1️⃣ IMPORT LIBRARIES
# ==========================================================
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

# ==========================================================
# 2️⃣ LOAD & PREPARE DATA
# ==========================================================

df = pd.read_csv('D:\Module 6 Project\Model Training Datasets\Train_df_SARIMAX_Model.csv')
df.sort_values(["NEIGHBOURHOOD", "YEAR"], inplace=True)
df.reset_index(drop=True, inplace=True)

df = df.rename(columns={
    "NEIGHBOURHOOD": "unique_id",
    "Total_Yearly_Crime_Count_per_Neighbourhood": "y",
    "YEAR": "year"
})

# Create datetime column (yearly frequency)
df["ds"] = pd.to_datetime(df["year"].astype(str) + "-01-01")

df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

# ==========================================================
# 3️⃣ SARIMA WALK-FORWARD CROSS VALIDATION
# ==========================================================

def sarima_cross_validation(data, h=2, n_windows=3,
                            order=(1,1,1),
                            seasonal_order=(0,0,0,0)):
    """
    h = forecast horizon (2 years)
    n_windows = rolling splits
    """

    results = []
    neighbourhoods = data["unique_id"].unique()

    for uid in neighbourhoods:

        sub = data[data["unique_id"] == uid].copy()
        sub = sub.set_index("ds")
        y = sub["y"]

        if len(y) < h + n_windows:
            continue

        for i in range(n_windows):

            split_point = len(y) - h*(n_windows - i)

            train = y.iloc[:split_point]
            test = y.iloc[split_point:split_point+h]

            try:
                model = SARIMAX(
                    train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )

                fitted = model.fit(disp=False)

                forecast = fitted.forecast(steps=h)

                temp = pd.DataFrame({
                    "unique_id": uid,
                    "ds": test.index,
                    "y": test.values,
                    "prediction": forecast.values
                })

                results.append(temp)

            except:
                continue

    return pd.concat(results).reset_index(drop=True)


# Run cross validation
cv_results = sarima_cross_validation(
    df,
    h=2,
    n_windows=3,
    order=(1,1,1),
    seasonal_order=(0,0,0,0)
)

# ==========================================================
# 4️⃣ GLOBAL METRICS
# ==========================================================

global_mae = mean_absolute_error(cv_results["y"], cv_results["prediction"])
global_mape = mean_absolute_percentage_error(cv_results["y"], cv_results["prediction"]) * 100

print(f"\nGlobal MAE: {global_mae:.3f}")
print(f"Global MAPE: {global_mape:.2f}%")

# ==========================================================
# 5️⃣ GLOBAL WMAPE
# ==========================================================

def global_wmape(df):
    return (np.sum(np.abs(df["y"] - df["prediction"])) /
            np.sum(df["y"])) * 100

print(f"Global wMAPE: {global_wmape(cv_results):.2f}%")

# ==========================================================
# 6️⃣ PER-NEIGHBOURHOOD METRICS
# ==========================================================

def per_neighbourhood_metrics(df):

    results = []

    grouped = df.groupby("unique_id")

    for uid, group in grouped:

        mae = mean_absolute_error(group["y"], group["prediction"])
        mape = mean_absolute_percentage_error(group["y"], group["prediction"]) * 100

        results.append({
            "unique_id": uid,
            "MAE": mae,
            "MAPE_%": mape
        })

    return pd.DataFrame(results).sort_values("MAPE_%", ascending=False)


print("\nPer-Neighbourhood Metrics:")
print(per_neighbourhood_metrics(cv_results))

# ==========================================================
# 7️⃣ FINAL MODEL (TRAIN ON FULL DATA)
#    FORECAST 2012–2013
# ==========================================================

def final_forecast(data, forecast_years=2,
                   order=(1,1,1),
                   seasonal_order=(0,0,0,0)):

    forecasts = []

    neighbourhoods = data["unique_id"].unique()

    for uid in neighbourhoods:

        sub = data[data["unique_id"] == uid].copy()
        sub = sub.set_index("ds")
        y = sub["y"]

        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(disp=False)

        forecast = fitted.forecast(steps=forecast_years)

        temp = pd.DataFrame({
            "unique_id": uid,
            "ds": forecast.index,
            "prediction": forecast.values
        })

        forecasts.append(temp)

    return pd.concat(forecasts).reset_index(drop=True)


final_predictions = final_forecast(df, forecast_years=2)

print("\nForecast for 2012–2013:")
print(final_predictions)

#save the sarima model for future use
import joblib
joblib.dump(final_forecast, 'final_SARIMA_forecast.pkl')