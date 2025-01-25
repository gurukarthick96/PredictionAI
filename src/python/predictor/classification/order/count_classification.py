# Step 0: Imports
from enum import Enum

import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder  # For Label Encoding
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Step 1: Prerequisites
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


class ClassificationType(Enum):
    LOGISTIC_REGRESSION = "Logistic Regression"
    RANDOM_FOREST = "Random Forest"
    DECISION_TREE = "Decision Tree"
    GRADIENT_BOOSTING = "Gradient Boosting"
    SVC = "Support Vector Classifier"
    KNN = "K-Nearest Neighbors"
    ADA_BOOST = "AdaBoost"
    XGBOOST = "XGBoost"


def build_classification_model(type):
    match type:
        case ClassificationType.LOGISTIC_REGRESSION:
            return LogisticRegression(max_iter=1000)
        case ClassificationType.RANDOM_FOREST:
            return RandomForestClassifier(n_estimators=100, random_state=42)
        case ClassificationType.DECISION_TREE:
            return DecisionTreeClassifier(random_state=42)
        case ClassificationType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(random_state=42)
        case ClassificationType.SVC:
            return SVC(kernel="rbf")  # Radial basis function kernel
        case ClassificationType.KNN:
            return KNeighborsClassifier(n_neighbors=5)
        case ClassificationType.ADA_BOOST:
            return AdaBoostClassifier(n_estimators=100, random_state=42)
        case ClassificationType.XGBOOST:
            return XGBClassifier(n_estimators=100, random_state=42)
    pass


# Step 2: Configurables
file_path = "../../../../resources/data/orders/orders_qa_120_days.json"

fields = ["day_of_week", "holiday"]
# fields = ['day_of_week', 'month', 'day_of_month', 'week_of_year', 'holiday']

# Step 3: Read Sample Data
df = pd.read_json(file_path)

# Feature engineering: Extract day of the week, month, etc.
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

# Step 4: Convert Target to Categorize (classification)
df["count_class"] = pd.qcut(df["count"], q=3, labels=["low", "medium", "high"])

# **Label Encoding**: Convert class labels to numeric (low -> 0, medium -> 1, high -> 2)
label_encoder = LabelEncoder()
df["count_class"] = label_encoder.fit_transform(df["count_class"])

# Features (input variables) and target (output variable)
X = df[fields]
y = df["count_class"]

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

# Step 6: Train and analyze performance of all classification models
results = []
for class_type in ClassificationType:
    model = build_classification_model(class_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results.append((class_type.value, round(accuracy * 100, 2), conf_matrix))

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Confusion Matrix"])
print(f"Classification Model Performance: \n{results_df}")

# Step 7: Choose the best classification model based on accuracy
best_model_type = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
best_model = build_classification_model(ClassificationType(best_model_type))

best_model.fit(X_train, y_train)

print(f"\nChosen Classification Model: {best_model_type}")

# Step 8: Predict on future data
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
future_df["predicted_count_class"] = best_model.predict(future_X)

# Step 9: Display original and future predictions
print(f"\nData Frame: \n{df}")

print("\nFuture Predictions: ")
future_df["predicted_count_class"] = label_encoder.inverse_transform(
    future_df["predicted_count_class"]
)
print(future_df[["date", "predicted_count_class"] + fields])

# Step 10: Evaluate the performance of the model on test data
y_pred = best_model.predict(X_test)
print(f"\nClassification Report: \n{classification_report(y_test, y_pred)}")
