ACIS Insurance Analytics

Overview

This repository supports AlphaCare Insurance Solutions (ACIS) in optimizing car insurance marketing and pricing strategies in South Africa. The project analyzes historical insurance claim data (Feb 2014 - Aug 2015) from insurance_data.txt to identify low-risk targets and refine premiums through EDA, hypothesis testing, and predictive modeling.

Project Structure

ACIS-Insurance-Analytics/
├── .dvc/
│   ├── config                  # DVC configuration
│   └── .gitignore              # DVC cache ignore
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD configuration
├── data/
│   ├── insurance_data.txt      # Raw dataset (DVC-tracked)
│   ├── insurance_data_cleaned.txt  # Cleaned dataset (DVC-tracked)
│   ├── *.dvc                   # DVC metadata
│   └── .gitignore              # Ignore datasets in Git
├── src/
│   ├── eda.py                  # EDA script
│   ├── preprocess_data.py      # Data cleaning script
│   ├── test_data.py            # Unit tests
│   ├── hypothesis_testing.py   # Hypothesis testing script
│   └── modeling.py             # Predictive modeling script
├── notebooks/
│   └── plots/                  # Visualizations
├── docs/
│   ├── eda_summary.md          # EDA findings
│   ├── hypothesis_testing_report.md  # Hypothesis testing results
│   ├── modeling_report.md      # Modeling results
│   ├── *.csv                   # Result CSVs
├── README.md                   # Project overview
├── requirements.txt            # Dependencies
└── .gitignore                  # Git ignore rules

Setup Instructions





Clone the Repository:

git clone https://github.com/<your-username>/ACIS-Insurance-Analytics.git
cd ACIS-Insurance-Analytics



Install Dependencies:

pip install -r requirements.txt



Set Up DVC:

dvc init
mkdir -p ~/dvc_storage/ACIS-Insurance-Analytics
dvc remote add -d localstorage ~/dvc_storage/ACIS-Insurance-Analytics
dvc pull



Generate Cleaned Dataset:

python src/preprocess_data.py
dvc add data/insurance_data_cleaned.txt
dvc push



Run EDA:

python src/eda.py



Run Hypothesis Tests:

python src/hypothesis_testing.py



Run Predictive Models:

python src/modeling.py



Run Tests:

pytest src/

Task 1: Git Setup and Exploratory Data Analysis





Deliverables: Git repository, eda.py, visualizations, eda_summary.md.



Insights: High premium variability, geographic trends.

Task 2: Data Version Control with DVC





Deliverables: DVC setup, tracked datasets, task-2 branch.



Features: Auditable data pipeline.

Task 3: A/B Hypothesis Testing





Deliverables: hypothesis_testing.py, results CSVs, hypothesis_testing_report.md, task-3 branch.



Findings: Provincial risk differences, no significant gender or zip code effects.

Task 4: Predictive Modeling





Objectives: Predict claim severity, claim probability, and optimize premiums.



Deliverables:





Data Preparation: Features (VehicleAge, PremiumPerSumInsured), one-hot encoding, 80:20 split.



Models: Linear Regression, Random Forests, XGBoost.



Evaluation: RMSE, R² for regression; accuracy, precision, recall, F1 for classification.



Interpretability: SHAP analysis identifying top features (e.g., CalculatedPremiumPerTerm, SumInsured).



Script: modeling.py.



Report: modeling_report.md with results and recommendations.



Branch: task-4 with 9+ commits.



Findings:





Severity: Linear Regression best (RMSE = 33,512, R² = 0.302), but modest performance.



Probability: XGBoost best (F1-score = 0.051), limited by class imbalance.



SHAP: CalculatedPremiumPerTerm, SumInsured, RegistrationYear drive severity.



Recommendations:





Adjust premiums for high SumInsured, older vehicles, urban areas.



Target low PremiumPerSumInsured policies.



Improve models with log-transformed targets or ensemble methods.

