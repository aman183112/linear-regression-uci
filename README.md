# 📘 Univariate Linear Regression 

A clean, beginner-friendly implementation of **Univariate Linear Regression**.

Dataset sourced directly from the **UCI Machine Learning Repository** (Student Performance, id=320).

---

## 📂 Project Structure

```
linear-regression-uci/
├── linear_regression.py   # Standalone Python script
├── notebook.ipynb         # Jupyter Notebook (recommended)
├── requirements.txt       # Python dependencies
├── plots/                 # Auto-generated output plots
│   ├── scatter_correlation.png
│   ├── loss_curve.png
│   ├── regression_line.png
│   └── residual_plot.png
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/aman183112/linear-regression-uci.git
cd linear-regression-uci
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

**Option A — Jupyter Notebook (recommended):**
```bash
jupyter notebook notebook.ipynb
```

**Option B — Python script:**
```bash
python linear_regression.py
```

---

## 🔍 What This Project Covers

| Step | Description |
|------|-------------|
| Data Loading | Fetched from UCI ML Repo using `ucimlrepo` |
| EDA | Shape, dtypes, missing values, statistics |
| Correlation Analysis | Auto-selects the feature most correlated with G3 |
| Feature Scaling | Standardisation (zero mean, unit variance) |
| Gradient Descent | Manual implementation |
| Evaluation | R² Score |
| Visualisations | Scatter plot, loss curve, regression line, residual plot |

---

## 📊 Dataset

- **Name:** Student Performance
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Target:** `G3` — Final student grade (0–20)

---

## 📦 Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `ucimlrepo`
- `jupyter`

---

## 📄 License

MIT License
