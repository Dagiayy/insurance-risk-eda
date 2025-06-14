
# Task 1 - Exploratory Data Analysis (EDA)

This project performs a comprehensive Exploratory Data Analysis (EDA) on a pipe-separated insurance dataset (`MachineLearningRating_v3.txt`) using Python and key libraries like Pandas, Matplotlib, and Seaborn. The objective is to uncover insights about insurance claims, premiums, customer and vehicle demographics, and data quality.

---

## 📁 Project Structure

```
.
├── data/
│   └── MachineLearningRating_v3.txt      # Input dataset (pipe-separated)
├── notebooks/
│   └── plots/                            # All generated visualizations
├── results/
│   └── eda_results.txt                   # Text summary of key EDA findings
├── task1_eda.py                          # Main Python script for EDA
└── README.md                             # This file
```

---

## ⚙️ Features of the Analysis

* **Data Cleaning & Imputation**

  * Converts date columns to datetime
  * Fills missing values in key fields like `TotalClaims`, `CustomValueEstimate`, `Gender`, and `Province`

* **Descriptive Statistics**

  * Summary of numerical columns
  * Missing value counts
  * Duplicate check on `UnderwrittenCoverID`

* **Univariate Analysis**

  * Histograms for numerical columns
  * Bar charts for categorical columns

* **Loss Ratio Analysis**

  * Calculates `LossRatio = TotalClaims / TotalPremium`
  * Analyzes by `make` and `CoverType`

* **Correlation Matrix**

  * Heatmap of correlations among key numeric variables

* **Outlier Detection**

  * Boxplots and counts of outliers for `TotalPremium` and `CustomValueEstimate`

* **Claim Behavior Analysis**

  * Monthly claim frequency and severity
  * Top vehicle makes with highest and lowest average claim amounts

* **Insightful Visualizations**

  * TotalPremium by `CoverType`
  * TotalPremium by vehicle `make`
  * Temporal trend of average premium

---

## 📊 Output

* **Visualizations** (in `notebooks/plots/`):

  * Histograms, bar charts, boxplots
  * Scatter plots and heatmaps
  * Time-series line charts

* **EDA Report** (in `results/eda_results.txt`):

  * Cleaned data summary
  * Descriptive stats, loss ratios, top trends, and outliers

---

## ✅ How to Run

1. Place your dataset at `data/MachineLearningRating_v3.txt`
2. Run the script:

   ```bash
   python eda/eda_summary.py
   ```

---

## 🧰 Requirements

Make sure the following Python packages are installed:

```bash
pip install pandas numpy matplotlib seaborn
```

---

## 📌 Notes

* If the dataset is missing, the script will exit with a message.
* All plots are saved in `notebooks/plots/`
* The summary report is saved in `results/eda_results.txt`

---
