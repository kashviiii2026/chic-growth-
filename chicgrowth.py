
# ============================================================
# CHIC MEDIA — SOCIAL GROWTH INTELLIGENCE
# Polynomial Regression & Model Evaluation
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

DATA_PATH = "chic_media_account_week.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 2. DATA CLEANING
# ------------------------------------------------------------

df["week_start"] = pd.to_datetime(
    df["week_start"],
    errors="coerce"
)

df["posts_per_week"] = pd.to_numeric(
    df["posts_per_week"],
    errors="coerce"
)

df["avg_engagement_rate"] = pd.to_numeric(
    df["avg_engagement_rate"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "week_start",
        "posts_per_week",
        "avg_engagement_rate"
    ]
)

df = df.sort_values("week_start").reset_index(drop=True)

print("\nCleaned dataset shape:", df.shape)

# ------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(
    df[
        [
            "posts_per_week",
            "avg_engagement_rate"
        ]
    ].describe()
)

print("\nCorrelation:")
print(
    df[
        [
            "posts_per_week",
            "avg_engagement_rate"
        ]
    ].corr()
)

# ------------------------------------------------------------
# 4. VISUALIZE ACTUAL DATA
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["posts_per_week"],
    df["avg_engagement_rate"],
    alpha=0.5
)

plt.xlabel("Posts Per Week")
plt.ylabel("Average Engagement Rate")
plt.title("Posting Frequency vs Engagement Rate")

plt.grid(True)
plt.tight_layout()
plt.savefig("actual_data_scatter.png")
plt.show()

# ------------------------------------------------------------
# 5. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

X = df[["posts_per_week"]]
y = df["avg_engagement_rate"]

# Chronological split: older observations for training,
# newer observations for testing.

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== TRAINING DATA ==========")

print("Total observations:", len(df))
print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

# ------------------------------------------------------------
# 6. POLYNOMIAL REGRESSION MODEL
# ------------------------------------------------------------

poly_model = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(
            degree=2,
            include_bias=False
        )
    ),
    (
        "regression",
        LinearRegression()
    )
])

poly_model.fit(X_train, y_train)

# ------------------------------------------------------------
# 7. PREDICTIONS
# ------------------------------------------------------------

y_train_pred = poly_model.predict(X_train)
y_test_pred = poly_model.predict(X_test)

# ------------------------------------------------------------
# 8. MODEL EVALUATION
# ------------------------------------------------------------

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

print("\n========== MODEL PERFORMANCE ==========")

print("\nTRAINING RESULTS")
print("R² Score:", round(train_r2, 4))
print("RMSE:", round(train_rmse, 4))
print("MAE:", round(train_mae, 4))

print("\nTESTING RESULTS")
print("R² Score:", round(test_r2, 4))
print("RMSE:", round(test_rmse, 4))
print("MAE:", round(test_mae, 4))

# ------------------------------------------------------------
# 9. POLYNOMIAL EQUATION
# ------------------------------------------------------------

regression = poly_model.named_steps["regression"]

intercept = regression.intercept_
coefficients = regression.coef_

b1 = coefficients[0]
b2 = coefficients[1]

print("\n========== POLYNOMIAL EQUATION ==========")

print(
    f"Engagement Rate = {intercept:.6f} "
    f"+ ({b1:.6f} × x) "
    f"+ ({b2:.6f} × x²)"
)

# ------------------------------------------------------------
# 10. OPTIMAL POSTING FREQUENCY
# ------------------------------------------------------------

if b2 < 0:

    optimal_frequency = -b1 / (2 * b2)

    min_frequency = X_train["posts_per_week"].min()
    max_frequency = X_train["posts_per_week"].max()

    print("\n========== OPTIMAL FREQUENCY ==========")

    if min_frequency <= optimal_frequency <= max_frequency:

        print(
            "Estimated optimal posting frequency:",
            round(optimal_frequency, 2),
            "posts per week"
        )

        optimal_prediction = poly_model.predict(
            pd.DataFrame({
                "posts_per_week": [optimal_frequency]
            })
        )[0]

        print(
            "Predicted engagement rate:",
            round(optimal_prediction, 4)
        )

    else:
        print(
            "Mathematical peak lies outside the training "
            "frequency range. Do not treat it as a "
            "reliable recommendation."
        )

else:
    optimal_frequency = None

    print(
        "\nNo interior maximum: the fitted curve is not "
        "concave downward."
    )

# ------------------------------------------------------------
# 11. PREDICT ENGAGEMENT AT DIFFERENT FREQUENCIES
# ------------------------------------------------------------

frequency_values = np.array(
    [2, 4, 6, 7, 8, 10, 12]
).reshape(-1, 1)

frequency_df = pd.DataFrame(
    frequency_values,
    columns=["posts_per_week"]
)

frequency_predictions = poly_model.predict(frequency_df)

prediction_table = pd.DataFrame({
    "Posts Per Week": frequency_values.flatten(),
    "Predicted Engagement Rate": frequency_predictions
})

print("\n========== FREQUENCY PREDICTIONS ==========")
print(prediction_table.to_string(index=False))

# ------------------------------------------------------------
# 12. POLYNOMIAL CURVE VISUALIZATION
# ------------------------------------------------------------

x_curve = np.linspace(
    X["posts_per_week"].min(),
    X["posts_per_week"].max(),
    200
).reshape(-1, 1)

x_curve_df = pd.DataFrame(
    x_curve,
    columns=["posts_per_week"]
)

y_curve = poly_model.predict(x_curve_df)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_train,
    y_train,
    alpha=0.35,
    label="Training Data"
)

plt.scatter(
    X_test,
    y_test,
    alpha=0.5,
    label="Testing Data"
)

plt.plot(
    x_curve,
    y_curve,
    linewidth=2,
    label="Polynomial Regression Curve"
)

if optimal_frequency is not None:
    if X_train["posts_per_week"].min() <= optimal_frequency <= X_train["posts_per_week"].max():

        plt.axvline(
            optimal_frequency,
            linestyle="--",
            label=f"Estimated Peak: {optimal_frequency:.2f}"
        )

plt.xlabel("Posts Per Week")
plt.ylabel("Average Engagement Rate")

plt.title(
    "Polynomial Regression: Posting Frequency vs Engagement"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("polynomial_regression.png")
plt.show()

# ------------------------------------------------------------
# 13. ACTUAL VS PREDICTED
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_test_pred,
    alpha=0.6
)

min_value = min(y_test.min(), y_test_pred.min())
max_value = max(y_test.max(), y_test_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Engagement Rate")
plt.ylabel("Predicted Engagement Rate")

plt.title("Actual vs Predicted Engagement Rate")

plt.grid(True)
plt.tight_layout()

plt.savefig("actual_vs_predicted.png")
plt.show()

# ------------------------------------------------------------
# 14. SAVE MODEL
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    poly_model,
    "models/polynomial_regression.pkl"
)

print("\nModel saved successfully!")
print("Location: models/polynomial_regression.pkl")

print("\n========== CHIC GROWTH ANALYSIS COMPLETE ==========")

