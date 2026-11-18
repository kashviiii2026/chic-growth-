
# ================================================================
# CHIC MEDIA — SOCIAL GROWTH INTELLIGENCE
# Complete Machine Learning Training Pipeline
#
# Model 1: Polynomial Regression (Baseline)
# Model 2: Multivariable Polynomial Regression + Ridge
#
# Target: avg_engagement_rate
# Dataset: chic_media_account_week.csv
# ================================================================


# ================================================================
# 1. IMPORT LIBRARIES
# ================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

warnings.filterwarnings("ignore")


# ================================================================
# 2. CONFIGURATION
# ================================================================

DATASET_PATH = "chic_media_account_week.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

TEST_SIZE = 0.20
POLYNOMIAL_DEGREE = 2

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 3. LOAD DATASET
# ================================================================

print("\n" + "=" * 65)
print("CHIC MEDIA — SOCIAL GROWTH INTELLIGENCE")
print("MACHINE LEARNING TRAINING PIPELINE")
print("=" * 65)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}\n"
        "Place the CSV file in the same folder as chicgrowth.py."
    )

df = pd.read_csv(DATASET_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())


# ================================================================
# 4. DATA CLEANING
# ================================================================

required_columns = [
    "week_start",
    "posts_per_week",
    "avg_engagement_rate",
    "follower_count",
    "account_age_months",
    "share_reels",
    "share_carousels",
    "share_images",
    "avg_caption_length",
    "avg_hashtag_count",
    "weekend_post_share",
    "industry"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["week_start"] = pd.to_datetime(
    df["week_start"],
    errors="coerce"
)

numeric_columns = [
    "posts_per_week",
    "avg_engagement_rate",
    "follower_count",
    "account_age_months",
    "share_reels",
    "share_carousels",
    "share_images",
    "avg_caption_length",
    "avg_hashtag_count",
    "weekend_post_share"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "week_start",
        "posts_per_week",
        "avg_engagement_rate",
        "industry"
    ]
)

df = df[
    (df["posts_per_week"] >= 0) &
    (df["avg_engagement_rate"].notna())
]

df = df.reset_index(drop=True)

if len(df) < 20:
    raise ValueError(
        "Not enough valid observations for model training."
    )

print("\nCleaned dataset shape:", df.shape)


# ================================================================
# 5. EXPLORATORY DATA ANALYSIS
# ================================================================

print("\n========== DESCRIPTIVE STATISTICS ==========")

print(
    df[
        ["posts_per_week", "avg_engagement_rate"]
    ].describe()
)

print("\nCorrelation:")

print(
    df[
        ["posts_per_week", "avg_engagement_rate"]
    ].corr()
)


# ================================================================
# 6. CHRONOLOGICAL TRAIN-TEST SPLIT
# ================================================================

# Sort by time to evaluate on later observations.

df = df.sort_values(
    "week_start"
).reset_index(drop=True)

split_index = int(len(df) * (1 - TEST_SIZE))

if split_index < 10 or split_index >= len(df):
    raise ValueError("Invalid train-test split.")

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("\n========== DATA SPLIT ==========")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "Training period:",
    train_df["week_start"].min(),
    "to",
    train_df["week_start"].max()
)

print(
    "Testing period:",
    test_df["week_start"].min(),
    "to",
    test_df["week_start"].max()
)


# ================================================================
# 7. EVALUATION FUNCTION
# ================================================================

def evaluate_model(name, y_train, train_pred, y_test, test_pred):

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, train_pred)
    )

    test_rmse = np.sqrt(
        mean_squared_error(y_test, test_pred)
    )

    train_mae = mean_absolute_error(
        y_train,
        train_pred
    )

    test_mae = mean_absolute_error(
        y_test,
        test_pred
    )

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    print("\nTRAINING RESULTS")
    print("R² Score:", round(train_r2, 4))
    print("RMSE:", round(train_rmse, 4))
    print("MAE:", round(train_mae, 4))

    print("\nTESTING RESULTS")
    print("R² Score:", round(test_r2, 4))
    print("RMSE:", round(test_rmse, 4))
    print("MAE:", round(test_mae, 4))

    return {
        "Model": name,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train MAE": train_mae,
        "Test MAE": test_mae
    }


# ================================================================
# 8. MODEL 1 — BASELINE POLYNOMIAL REGRESSION
# ================================================================

print("\n\n========== MODEL 1: BASELINE ==========")

X_train = train_df[["posts_per_week"]]
X_test = test_df[["posts_per_week"]]

y_train = train_df["avg_engagement_rate"]
y_test = test_df["avg_engagement_rate"]

baseline_model = Pipeline([
    (
        "polynomial",
        PolynomialFeatures(
            degree=POLYNOMIAL_DEGREE,
            include_bias=False
        )
    ),
    (
        "regression",
        LinearRegression()
    )
])

print("\nTraining baseline model...")

baseline_model.fit(X_train, y_train)

baseline_train_pred = baseline_model.predict(X_train)
baseline_test_pred = baseline_model.predict(X_test)

baseline_metrics = evaluate_model(
    "BASELINE POLYNOMIAL REGRESSION",
    y_train,
    baseline_train_pred,
    y_test,
    baseline_test_pred
)


# ================================================================
# 9. BASELINE POLYNOMIAL EQUATION
# ================================================================

baseline_regression = baseline_model.named_steps["regression"]

baseline_intercept = baseline_regression.intercept_
baseline_coefficients = baseline_regression.coef_

b1 = baseline_coefficients[0]
b2 = baseline_coefficients[1]

print("\n========== BASELINE POLYNOMIAL EQUATION ==========")

print(
    f"Engagement Rate = {baseline_intercept:.6f} "
    f"+ ({b1:.6f} × x) "
    f"+ ({b2:.6f} × x²)"
)


# ================================================================
# 10. BASELINE OPTIMAL POSTING FREQUENCY
# ================================================================

print("\n========== BASELINE OPTIMAL FREQUENCY ==========")

baseline_optimal_frequency = None

if b2 < 0:

    candidate = -b1 / (2 * b2)

    minimum_frequency = X_train["posts_per_week"].min()
    maximum_frequency = X_train["posts_per_week"].max()

    if minimum_frequency <= candidate <= maximum_frequency:

        baseline_optimal_frequency = candidate

        prediction = baseline_model.predict(
            pd.DataFrame({
                "posts_per_week": [candidate]
            })
        )[0]

        print(
            "Estimated optimal frequency:",
            round(candidate, 2),
            "posts per week"
        )

        print(
            "Predicted engagement rate:",
            round(prediction, 4)
        )

    else:
        print("Estimated peak is outside the training range.")

else:
    print("No interior maximum identified.")


# ================================================================
# 11. MODEL 2 — MULTIVARIABLE FEATURE SELECTION
# ================================================================

print("\n\n========== MODEL 2: MULTIVARIABLE ==========")

# Content shares represent the proportion of each content type.
# They avoid directly duplicating total posting frequency.

numeric_features = [
    "posts_per_week",
    "share_reels",
    "share_carousels",
    "share_images",
    "follower_count",
    "account_age_months",
    "avg_caption_length",
    "avg_hashtag_count",
    "weekend_post_share"
]

categorical_features = [
    "industry"
]

target = "avg_engagement_rate"

X_train_multi = train_df[
    numeric_features + categorical_features
]

X_test_multi = test_df[
    numeric_features + categorical_features
]

y_train_multi = train_df[target]
y_test_multi = test_df[target]

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

print("\nTarget:", target)


# ================================================================
# 12. MULTIVARIABLE PREPROCESSING
# ================================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "polynomial",
        PolynomialFeatures(
            degree=POLYNOMIAL_DEGREE,
            include_bias=False
        )
    ),
    (
        "polynomial_scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ================================================================
# 13. BUILD MULTIVARIABLE MODEL
# ================================================================

multi_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "regression",
        Ridge(alpha=1.0)
    )
])


# ================================================================
# 14. TRAIN MULTIVARIABLE MODEL
# ================================================================

print("\nTraining multivariable model...")

multi_model.fit(
    X_train_multi,
    y_train_multi
)

print("Multivariable model training completed!")


# ================================================================
# 15. MULTIVARIABLE PREDICTIONS
# ================================================================

multi_train_pred = multi_model.predict(
    X_train_multi
)

multi_test_pred = multi_model.predict(
    X_test_multi
)


# ================================================================
# 16. MULTIVARIABLE MODEL EVALUATION
# ================================================================

multi_metrics = evaluate_model(
    "MULTIVARIABLE POLYNOMIAL REGRESSION + RIDGE",
    y_train_multi,
    multi_train_pred,
    y_test_multi,
    multi_test_pred
)


# ================================================================
# 17. COMPARE BOTH MODELS
# ================================================================

print("\n\n========== MODEL COMPARISON ==========")

comparison = pd.DataFrame([
    baseline_metrics,
    multi_metrics
])

print(
    comparison.round(4).to_string(index=False)
)

comparison.to_csv(
    f"{OUTPUT_DIR}/model_comparison.csv",
    index=False
)


# ================================================================
# 18. ACCOUNT-SPECIFIC POSTING FREQUENCY EXPERIMENT
# ================================================================

print("\n========== ACCOUNT-SPECIFIC FREQUENCY ==========")

# Use the first available test observation as an example profile.
# Change posting frequency while keeping other features constant.

sample_account = X_test_multi.iloc[[0]].copy()

observed_min = int(
    train_df["posts_per_week"].min()
)

observed_max = int(
    train_df["posts_per_week"].max()
)

frequency_results = []

for frequency in range(observed_min, observed_max + 1):

    scenario = sample_account.copy()

    scenario["posts_per_week"] = frequency

    predicted_engagement = multi_model.predict(
        scenario
    )[0]

    frequency_results.append({
        "posts_per_week": frequency,
        "predicted_engagement_rate": predicted_engagement
    })


frequency_df = pd.DataFrame(frequency_results)

print(
    frequency_df.round(4).to_string(index=False)
)

best_row = frequency_df.loc[
    frequency_df["predicted_engagement_rate"].idxmax()
]

print("\nBest tested posting frequency:")

print(
    int(best_row["posts_per_week"]),
    "posts per week"
)

print(
    "Predicted engagement:",
    round(best_row["predicted_engagement_rate"], 4)
)

frequency_df.to_csv(
    f"{OUTPUT_DIR}/account_frequency_experiment.csv",
    index=False
)


# ================================================================
# 19. VISUALIZATION — BASELINE REGRESSION
# ================================================================

X_range = np.linspace(
    X_train["posts_per_week"].min(),
    X_train["posts_per_week"].max(),
    300
)

X_range_df = pd.DataFrame({
    "posts_per_week": X_range
})

y_range_pred = baseline_model.predict(
    X_range_df
)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_train["posts_per_week"],
    y_train,
    alpha=0.5,
    label="Training Data"
)

plt.scatter(
    X_test["posts_per_week"],
    y_test,
    alpha=0.5,
    label="Testing Data"
)

plt.plot(
    X_range,
    y_range_pred,
    linewidth=2,
    label="Polynomial Regression"
)

if baseline_optimal_frequency is not None:

    plt.axvline(
        baseline_optimal_frequency,
        linestyle="--",
        label="Estimated Peak"
    )

plt.title("Posting Frequency vs Engagement Rate")
plt.xlabel("Posts Per Week")
plt.ylabel("Average Engagement Rate")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    f"{OUTPUT_DIR}/baseline_polynomial_regression.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 20. VISUALIZATION — ACTUAL VS PREDICTED
# ================================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_multi,
    multi_test_pred,
    alpha=0.6
)

min_value = min(
    y_test_multi.min(),
    multi_test_pred.min()
)

max_value = max(
    y_test_multi.max(),
    multi_test_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Multivariable Model: Actual vs Predicted")
plt.xlabel("Actual Engagement Rate")
plt.ylabel("Predicted Engagement Rate")
plt.grid(alpha=0.3)

plt.savefig(
    f"{OUTPUT_DIR}/multivariable_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 21. VISUALIZATION — FREQUENCY EXPERIMENT
# ================================================================

plt.figure(figsize=(9, 5))

plt.plot(
    frequency_df["posts_per_week"],
    frequency_df["predicted_engagement_rate"],
    marker="o"
)

plt.title("Account-Specific Posting Frequency Experiment")
plt.xlabel("Posts Per Week")
plt.ylabel("Predicted Engagement Rate")
plt.grid(alpha=0.3)

plt.savefig(
    f"{OUTPUT_DIR}/account_frequency_experiment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ================================================================
# 22. SAVE TRAINED MODELS
# ================================================================

joblib.dump(
    baseline_model,
    f"{MODEL_DIR}/baseline_polynomial_model.pkl"
)

joblib.dump(
    multi_model,
    f"{MODEL_DIR}/multivariable_engagement_model.pkl"
)

print("\nBoth models saved successfully!")


# ================================================================
# 23. SAVE TEST PREDICTIONS
# ================================================================

test_predictions = test_df[
    ["week_start", "account_id", "industry"]
].copy()

test_predictions["actual_engagement_rate"] = y_test_multi.values

test_predictions["predicted_engagement_rate"] = multi_test_pred

test_predictions.to_csv(
    f"{OUTPUT_DIR}/test_predictions.csv",
    index=False
)


# ================================================================
# 24. SAVE MODEL METRICS
# ================================================================

comparison.to_csv(
    f"{OUTPUT_DIR}/model_metrics.csv",
    index=False
)


# ================================================================
# 25. FINAL SUMMARY
# ================================================================

print("\n" + "=" * 65)
print("CHIC MEDIA — TRAINING COMPLETE")
print("=" * 65)

print("\nMODEL 1: BASELINE")
print("Testing R²:", round(baseline_metrics["Test R2"], 4))
print("Testing RMSE:", round(baseline_metrics["Test RMSE"], 4))

print("\nMODEL 2: MULTIVARIABLE")
print("Testing R²:", round(multi_metrics["Test R2"], 4))
print("Testing RMSE:", round(multi_metrics["Test RMSE"], 4))
print("Testing MAE:", round(multi_metrics["Test MAE"], 4))

print("\nOUTPUT FILES:")
print("models/baseline_polynomial_model.pkl")
print("models/multivariable_engagement_model.pkl")
print("outputs/model_comparison.csv")
print("outputs/model_metrics.csv")
print("outputs/test_predictions.csv")
print("outputs/account_frequency_experiment.csv")
print("outputs/baseline_polynomial_regression.png")
print("outputs/multivariable_actual_vs_predicted.png")
print("outputs/account_frequency_experiment.png")

print("\nCHIC MEDIA ML ENGINE IS READY FOR THE NEXT DEVELOPMENT PHASE!")
print("=" * 65)
