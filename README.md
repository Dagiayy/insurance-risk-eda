ACIS Insurance Analytics

Overview

This repository supports AlphaCare Insurance Solutions (ACIS) in optimizing car insurance marketing and pricing strategies in South Africa. The project analyzes historical insurance claim data (Feb 2014 - Aug 2015) from a pipe-separated text file (insurance_data.txt) to identify low-risk targets and refine premiums through Exploratory Data Analysis (EDA), hypothesis testing, and predictive modeling. Task 1 established version control and EDA, while Task 2 implements Data Version Control (DVC) for reproducible data pipelines.

Project Structure

ACIS-Insurance-Analytics/
├── .dvc/
│   ├── config                  # DVC configuration
│   └── .gitignore              # DVC cache ignore
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD configuration
├── data/
│   ├── insurance_data.txt      # Pipe-separated dataset (tracked by DVC)
│   ├── insurance_data.txt.dvc  # DVC metadata
│   └── .gitignore              # Ignore dataset in Git
├── src/
│   ├── eda.py                  # EDA script for Task 1
│   └── test_data.py            # Unit tests for data loading
├── notebooks/
│   └── plots/                  # Directory for EDA visualizations
│       ├── hist_*.png          # Histograms for numerical variables
│       ├── bar_*.png           # Bar charts for categorical variables
│       ├── box_*.png           # Box plots for outlier detection
│       ├── correlation_matrix.png
│       ├── scatter_premium_claims_zip.png
│       ├── premium_by_covertype.png
│       ├── premium_by_make.png
│       └── temporal_premium_trend.png
├── docs/
│   └── eda_summary.md          # Summary of EDA findings
├── README.md                   # Project overview and setup instructions
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files to ignore in Git

Setup Instructions





Clone the Repository:

git clone https://github.com/<your-username>/ACIS-Insurance-Analytics.git
cd ACIS-Insurance-Analytics



Install Dependencies:

pip install -r requirements.txt



Set Up DVC:





Initialize DVC (if not already done):

dvc init



Configure local remote storage:

mkdir -p ~/dvc_storage/ACIS-Insurance-Analytics
dvc remote add -d localstorage ~/dvc_storage/ACIS-Insurance-Analytics



Pull dataset:

dvc pull



Run EDA:

python src/eda.py





Outputs: Console logs (data types, statistics, etc.) and plots in notebooks/plots/.



Run Tests:

pytest src/

Task 1: Git Setup and Exploratory Data Analysis

Objectives





Git and GitHub Setup: Initialize a Git repository, configure GitHub Actions for CI/CD, and establish a branching workflow.



EDA: Perform comprehensive analysis on insurance_data.txt to understand data distributions, relationships, and trends.

Deliverables





Git Repository:





Initialized with git init and pushed to GitHub.



Configured with .gitignore to exclude data files and outputs.



CI/CD pipeline (ci.yml) for linting (flake8) and testing (pytest).



Branch task-1 with 16+ commits (minimum 3 per day), merged into main via PR.



EDA:





Script: src/eda.py loads pipe-separated data, performs summarization, quality checks, univariate/bivariate analyses, outlier detection, and visualizations.



Data Summarization: Descriptive statistics for numerical features (e.g., TotalPremium, SumInsured).



Data Quality: Handled missing values (e.g., median imputation for CustomValueEstimate) and duplicates (883,566 in UnderwrittenCoverID).



Univariate Analysis: Histograms for numerical variables, bar charts for categorical variables (e.g., Province, CoverType).



Bivariate Analysis: Loss Ratio by make and CoverType, correlation matrix, scatter plot of TotalPremium vs. TotalClaims by PostalCode.



Geographic Trends: Average TotalPremium by MainCrestaZone and CoverType, top vehicle makes per zone.



Outlier Detection: Identified 209,042 outliers in TotalPremium and 217,880 in CustomValueEstimate using IQR method.



Temporal Trends: Claim frequency (0.000206 to 0.003710) and severity (3,094 to 46,355 Rand) by month.



Vehicle Makes: Highest claim amounts for Polarsun (125,197 Rand), lowest for Ford (2,040 Rand).



Visualizations:





Bar plot: Average TotalPremium by CoverType.



Box plot: TotalPremium by vehicle make.



Line plot: Average TotalPremium over time.



Documentation: docs/eda_summary.md summarizes findings and recommendations.



Tests: src/test_data.py verifies data loading and column presence.

Key Insights





TotalPremium varies significantly by CoverType (e.g., Own Damage: high premiums, Windscreen: low).



High Loss Ratios for certain makes (e.g., Marcopolo: 1.57e12) due to outlier claims.



Rand East and Johannesburg have higher premiums than other zones.

Task 2: Data Version Control with DVC

Objectives





Establish a reproducible data pipeline using DVC to version-control insurance_data.txt.



Ensure data inputs are auditable for regulatory compliance in insurance.

Deliverables





DVC Installation:





Installed via pip install dvc and added to requirements.txt.



DVC Initialization:





Initialized with dvc init, creating .dvc/ directory.



Local Remote Storage:





Configured at ~/dvc_storage/ACIS-Insurance-Analytics using dvc remote add.



Data Tracking:





Tracked data/insurance_data.txt with dvc add, creating insurance_data.txt.dvc.



Created a second version of the dataset (e.g., added a dummy row).



Version Control:





Committed .dvc files and updated .gitignore to Git.



Branch task-2 created with 7+ commits.



Data Push:





Pushed dataset versions to local remote with dvc push.

Key Commands





Track new data version:

dvc add data/insurance_data.txt
git add data/insurance_data.txt.dvc
git commit -m "Updated dataset version"



Push to remote:

dvc push



Retrieve specific version:

git checkout <commit-hash>
dvc pull


Final Submission: June 17, 2025, 8:00 PM UTC