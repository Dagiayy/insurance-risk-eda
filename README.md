ACIS Insurance Analytics
Overview
This repository supports AlphaCare Insurance Solutions (ACIS) in optimizing car insurance marketing and pricing strategies in South Africa. The project analyzes historical insurance claim data (Feb 2014 - Aug 2015) from a pipe-separated text file (insurance_data.txt) to identify low-risk targets and refine premiums through Exploratory Data Analysis (EDA), hypothesis testing, and predictive modeling. Task 1 established version control and EDA, Task 2 implemented Data Version Control (DVC), and Task 3 conducted A/B hypothesis testing.
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
│   └── hypothesis_testing.py   # Hypothesis testing script
├── notebooks/
│   └── plots/                  # EDA visualizations
├── docs/
│   ├── eda_summary.md          # EDA findings
│   ├── hypothesis_testing_report.md  # Hypothesis testing results
│   ├── hypothesis_test_results.csv   # Test results
│   └── equivalence_check_results.csv # Equivalence checks
├── README.md                   # Project overview
├── requirements.txt            # Dependencies
└── .gitignore                  # Git ignore rules

Setup Instructions

Clone the Repository:git clone https://github.com/<your-username>/ACIS-Insurance-Analytics.git
cd ACIS-Insurance-Analytics


Install Dependencies:pip install -r requirements.txt


Set Up DVC:dvc init
mkdir -p ~/dvc_storage/ACIS-Insurance-Analytics
dvc remote add -d localstorage ~/dvc_storage/ACIS-Insurance-Analytics
dvc pull


Generate Cleaned Dataset:python src/preprocess_data.py
dvc add data/insurance_data_cleaned.txt
dvc push


Run EDA:python src/eda.py


Run Hypothesis Tests:python src/hypothesis_testing.py


Run Tests:pytest src/



Task 1: Git Setup and Exploratory Data Analysis

Objectives: Initialize Git repository, configure CI/CD, perform EDA.
Deliverables:
Git repository with task-1 branch (merged into main).
EDA script (eda.py) analyzing insurance_data.txt.
Visualizations and documentation (eda_summary.md).


Insights: High premium variability, significant outliers, geographic trends.

Task 2: Data Version Control with DVC

Objectives: Establish reproducible data pipeline.
Deliverables:
DVC setup with local remote (~/dvc_storage/ACIS-Insurance-Analytics).
Tracked datasets: insurance_data.txt, insurance_data_cleaned.txt.
task-2 branch with 10+ commits.


Features: Versioned data for auditability.

Task 3: A/B Hypothesis Testing

Objectives: Validate risk drivers (provinces, zip codes, gender) using Claim Frequency, Claim Severity, and Margin.
Deliverables:
Metrics: Claim Frequency, Claim Severity, Margin.
Segmentation: Gauteng vs. Western Cape, zip codes 2000 vs. 8000, Male vs. Female.
Tests: Chi-squared for frequency, t-tests for severity and margin.
Script: hypothesis_testing.py using insurance_data_cleaned.txt.
Report: hypothesis_testing_report.md with actual results.
Branch: task-3 with 5+ commits.


Findings:
Rejected H₀₁: Gauteng has 55% higher claim frequency; Western Cape has 26% higher severity.
Failed to reject H₀₂, H₀₃, H₀₄: No significant differences for zip codes or gender, but monitor zip code 8000 severity and negative margins.


Recommendations:
Increase premiums in Gauteng (10-15%) and Western Cape (5-10%).
Monitor zip code 8000 claim severity.
Review pricing to address negative margins.
Maintain gender-neutral pricing.

