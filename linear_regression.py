# =========================================================
# UNIVARIATE LINEAR REGRESSION FROM SCRATCH
# Dataset: Student Performance (UCI ML Repository, id=320)
# Using ONLY numpy, pandas, matplotlib, ucimlrepo
# =========================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


# =========================
# 2. LOAD DATASET
# =========================

# Fetch Student Performance dataset from UCI Repository
student_performance = fetch_ucirepo(id=320)

# Features and target
X_data = student_performance.data.features
y_data = student_performance.data.targets

# Combine into one dataframe for easier EDA
df = pd.concat([X_data, y_data], axis=1)


# =========================
# 3. DATASET INFORMATION
# =========================

print("Dataset Metadata:\n")
print(student_performance.metadata)

print("\nVariable Information:\n")
print(student_performance.variables)


# =========================
# 4. BASIC EDA
# =========================

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:\n", df.columns.tolist())

print("\nMissing Values:\n", df.isnull().sum())

print("\nDataset Information:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())


# =========================
# 5. SELECT NUMERIC COLUMNS
# =========================

numeric_columns = df.select_dtypes(include=np.number)


# =========================
# 6. CORRELATION ANALYSIS
# =========================

# G3 is the final student grade (target)
correlation = numeric_columns.corr()['G3']

print("\nCorrelation With G3:\n")
print(correlation)

correlation = correlation.drop('G3')
correlation = correlation.sort_values(key=abs, ascending=False)

FEATURE = correlation.index[0]
print("\nSelected Feature:", FEATURE)


# =========================
# 7. VISUALIZE CORRELATION
# =========================

plt.figure(figsize=(6, 4))
plt.scatter(df[FEATURE], df['G3'], alpha=0.6, color='steelblue',
            edgecolors='k', linewidths=0.4)
plt.xlabel(FEATURE, fontsize=12)
plt.ylabel("G3", fontsize=12)
plt.title(f"{FEATURE} vs G3", fontsize=14)
plt.tight_layout()
plt.savefig("plots/scatter_correlation.png", dpi=150)
plt.show()


# =========================
# 8. PREPARE DATA
# =========================

X = df[FEATURE].values.astype(float)
y = df['G3'].values.astype(float)
n = len(X)


# =========================
# 9. FEATURE SCALING
# =========================

X_mean = np.mean(X)
X_std  = np.std(X)
X = (X - X_mean) / X_std


# =========================
# 10. INITIALIZE PARAMETERS
# =========================

# Linear equation: y = wx + b
w = 0.0
b = 0.0

learning_rate = 0.01
iterations    = 1000
loss_history  = []


# =========================
# 11. GRADIENT DESCENT
# =========================

for i in range(iterations):
    predictions = w * X + b
    error       = predictions - y

    dw = (2/n) * np.sum(error * X)
    db = (2/n) * np.sum(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    loss = np.mean(error ** 2)
    loss_history.append(loss)


# =========================
# 12. FINAL PARAMETERS
# =========================

print(f"\nFinal Weight (w): {w:.4f}")
print(f"Final Bias   (b): {b:.4f}")


# =========================
# 13. MAKE PREDICTIONS
# =========================

predictions = w * X + b


# =========================
# 14. CALCULATE R² SCORE
# =========================

residual_error = np.sum((y - predictions) ** 2)
total_error    = np.sum((y - np.mean(y)) ** 2)
r2_score       = 1 - (residual_error / total_error)

print(f"\nR² Score: {round(r2_score, 4)}")


# =========================
# 15. PLOT LOSS CURVE
# =========================

plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='crimson', linewidth=1.5)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.title("Loss vs Iterations", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/loss_curve.png", dpi=150)
plt.show()


# =========================
# 16. PLOT REGRESSION LINE
# =========================

sort_idx = np.argsort(X)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.6, color='steelblue',
            edgecolors='k', linewidths=0.4, label="Actual Data")
plt.plot(X[sort_idx], predictions[sort_idx],
         color='red', linewidth=2, label="Regression Line")
plt.xlabel(f"{FEATURE} (scaled)", fontsize=12)
plt.ylabel("G3", fontsize=12)
plt.title("Linear Regression Fit", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/regression_line.png", dpi=150)
plt.show()


# =========================
# 17. RESIDUAL PLOT
# =========================

residuals = y - predictions

plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals, alpha=0.6, color='darkorange',
            edgecolors='k', linewidths=0.4)
plt.axhline(y=0, color='black', linewidth=1.2, linestyle='--')
plt.xlabel("Predicted Values", fontsize=12)
plt.ylabel("Residuals", fontsize=12)
plt.title("Residual Plot", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("plots/residual_plot.png", dpi=150)
plt.show()


# =========================
# 18. FINAL CONCLUSION
# =========================

print("\nModel Training Completed Successfully!")
print("The model used one feature to predict final student grades.")
print("Gradient Descent was implemented manually using loops.")
