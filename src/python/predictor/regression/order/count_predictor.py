# Step 0: Imports
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Step 1: Prerequisites
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)


class RegressionType(Enum):
    LINEAR = "Linear Regression"
    RANDOM_FOREST = "Random Forest"
    DECISION_TREE = "Decision Tree"
    GRADIENT_BOOSTING = "Gradient Boosting"
    SVR = "Support Vector Regressor"
    KNN = "K-Nearest Neighbors"
    ELASTIC_NET = "ElasticNet"
    ADA_BOOST = "AdaBoost"
    XGBOOST = "XGBoost"


def build_regression_model(type):
    match type:
        case RegressionType.LINEAR:
            return LinearRegression()
        case RegressionType.RANDOM_FOREST:
            return RandomForestRegressor(n_estimators=100, random_state=42)
        case RegressionType.DECISION_TREE:
            return DecisionTreeRegressor(random_state=42)
        case RegressionType.GRADIENT_BOOSTING:
            return GradientBoostingRegressor(random_state=42)
        case RegressionType.SVR:
            return SVR(kernel="rbf")  # Radial basis function kernel
        case RegressionType.KNN:
            return KNeighborsRegressor(n_neighbors=5)
        case RegressionType.ELASTIC_NET:
            return ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        case RegressionType.ADA_BOOST:
            return AdaBoostRegressor(n_estimators=100, random_state=42)
        case RegressionType.XGBOOST:
            return XGBRegressor(n_estimators=100, random_state=42)
    pass


# Step 2: Configurables
file_path = "../../../../resources/data/orders/orders_apac_30_days.json"

fields = ["day_of_week", "holiday"]
# fields = ['day_of_week', 'month', 'day_of_month', 'week_of_year', 'holiday']

# regressionType = RegressionType.LINEAR
# print(f'Regression Model: {regressionType.value}\n')

# Step 3: Read Sample Data
df = pd.read_json(file_path)

# below conversion needed if loaded from csv
# df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Step 4: Feature engineering: Extract day of the week, month, etc.
if "day_of_week" in fields:
    df["day_of_week"] = df["date"].dt.dayofweek  # Monday=0, Sunday=6
if "month" in fields:
    df["month"] = df["date"].dt.month  # 1 to 12
if "day_of_month" in fields:
    df["day_of_month"] = df["date"].dt.day  # 1 to 31
if "week_of_year" in fields:
    df["week_of_year"] = df["date"].dt.isocalendar().week  # 1 to 52
if "holiday" in fields:
    df["holiday"] = df["date"].dt.dayofweek.isin([5, 6])  # True or False

# Step 5: Define Target and features
X = df[fields]
y = df["count"]

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 7: Train and analyze performance of all regression model
results = []
for reg_type in RegressionType:
    model = build_regression_model(reg_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results.append((reg_type.value, round(rmse, 2), round(mse, 2)))

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MSE"])
print(f"Regression Model Performance: \n{results_df}")

# Step 8: Choose the best regression model
best_model_type = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
best_model = build_regression_model(RegressionType(best_model_type))

best_model.fit(X_train, y_train)

print(f"\nChosen Regression Model: {best_model_type}")

# Step 9: Predict on future data
future_dates = pd.date_range(start="2025-01-25", end="2025-02-24")
future_df = pd.DataFrame(
    {
        "date": future_dates,
        **({"day_of_week": future_dates.dayofweek} if "day_of_week" in fields else {}),
        **({"month": future_dates.month} if "month" in fields else {}),
        **({"day_of_month": future_dates.day} if "day_of_month" in fields else {}),
        **(
            {"week_of_year": future_dates.isocalendar().week}
            if "week_of_year" in fields
            else {}
        ),
        **(
            {"holiday": future_dates.dayofweek.isin([5, 6])}
            if "holiday" in fields
            else {}
        ),
    }
)

future_X = future_df[fields]
future_df["predicted_count"] = best_model.predict(future_X)

future_df["predicted_count"] = (
    future_df["predicted_count"].clip(lower=0).round().astype(int)
)

# Step 10: Display original and future predictions
print(f"\nData Frame: \n{df}")

print("\nFuture Predictions: ")
print(future_df[["date", "predicted_count"] + fields])
