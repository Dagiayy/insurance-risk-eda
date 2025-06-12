# 🚗 Insurance Risk & Profitability Analysis

This project performs **Exploratory Data Analysis (EDA)**, **statistical modeling**, and **machine learning** on historical insurance data provided by AlphaCare Insurance Solutions. The primary goal is to uncover patterns in **risk**, **claim behavior**, and **profitability**, and to propose **data-driven recommendations** to improve policy and premium strategies.

---

## 📦 Dataset Overview

- **Time Period:** February 2014 – August 2015  
- **Dataset Location:** `data/insurance_data.csv`  
- **Rows:** Individual insurance policies  
- **Columns include:**
  - **Client Details:** Gender, Marital Status, Citizenship, etc.
  - **Location Info:** Province, PostalCode, SubCrestaZone, etc.
  - **Vehicle Info:** Make, Model, Year, Power, Type, etc.
  - **Plan Info:** Premiums, SumInsured, CoverType, etc.
  - **Payments & Claims:** TotalPremium, TotalClaims

---

## 📊 Tasks Breakdown

### 1. Git & GitHub Setup
- Git repository with CI/CD using GitHub Actions
- Branching strategy (`task-1`, `main`)
- Regular descriptive commits

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics
- Univariate & bivariate plots
- Loss Ratio analysis across gender, province, and vehicle type
- Outlier detection (box plots)
- Trends over time and geography

### 3. A/B Hypothesis Testing
Testing the following null hypotheses:
- No risk differences across provinces or zipcodes
- No significant margin (profit) differences between zipcodes
- No significant risk difference between genders

### 4. Statistical & Machine Learning Modeling
- Linear regression per ZipCode to predict **TotalClaims**
- ML model to predict **optimal premium values** using:
  - Vehicle features
  - Owner demographics
  - Location-based features
- Feature importance analysis

---

## 🛠 Technologies Used

- **Language:** Python 3.10+
- **Libraries:**  
  - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`
- **CI/CD:** GitHub Actions
- **Version Control:** Git + GitHub

---

## 🧪 How to Run the Project

```bash
# Clone the repo
git clone https://github.com/Dagiayy/insurance-risk-eda.git
cd insurance-risk-eda

# Activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run EDA summary script
python eda/eda_summary.py
