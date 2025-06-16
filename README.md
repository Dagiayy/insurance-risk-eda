
---

# **ACIS Insurance Analytics**

This repository supports **AlphaCare Insurance Solutions (ACIS)** in optimizing car insurance marketing and pricing strategies in South Africa. The project analyzes historical insurance claim data from February 2014 to August 2015, stored in a pipe-separated text file (`insurance_data.txt`), to identify low-risk targets and refine premiums through **Exploratory Data Analysis (EDA)**, **hypothesis testing**, and **predictive modeling**.

The project is structured into three key tasks:
- **Task 1:** Git setup and Exploratory Data Analysis (EDA).
- **Task 2:** Implementation of Data Version Control (DVC) for reproducibility.
- **Task 3:** A/B hypothesis testing to validate risk drivers and refine pricing strategies.

---

## **Project Structure**

```
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
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/ACIS-Insurance-Analytics.git
cd ACIS-Insurance-Analytics
```

### **2. Install Dependencies**
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### **3. Set Up DVC**
Initialize DVC and configure a local remote storage for versioning datasets:
```bash
dvc init
mkdir -p ~/dvc_storage/ACIS-Insurance-Analytics
dvc remote add -d localstorage ~/dvc_storage/ACIS-Insurance-Analytics
dvc pull
```

### **4. Generate Cleaned Dataset**
Run the data preprocessing script to clean the raw dataset:
```bash
python src/preprocess_data.py
```
Track the cleaned dataset with DVC:
```bash
dvc add data/insurance_data_cleaned.txt
dvc push
```

### **5. Run Exploratory Data Analysis (EDA)**
Generate insights and visualizations by running the EDA script:
```bash
python src/eda.py
```
Findings are documented in `docs/eda_summary.md`.

### **6. Run Hypothesis Tests**
Validate risk drivers (e.g., provinces, zip codes, gender) using the hypothesis testing script:
```bash
python src/hypothesis_testing.py
```
Results are summarized in `docs/hypothesis_testing_report.md`.

### **7. Run Tests**
Ensure code quality and functionality by running unit tests:
```bash
pytest src/
```

---

## **Task Breakdown**

### **Task 1: Git Setup and Exploratory Data Analysis**
**Objective:** Initialize a Git repository, configure CI/CD pipelines, and perform EDA on the raw dataset.

**Deliverables:**
- Git repository with a `task-1` branch merged into `main`.
- EDA script (`src/eda.py`) analyzing `insurance_data.txt`.
- Visualizations and documentation (`docs/eda_summary.md`).

**Key Insights:**
- High variability in premiums across different regions.
- Significant outliers in claim severity.
- Geographic trends indicating higher claim frequencies in specific provinces.

---

### **Task 2: Data Version Control with DVC**
**Objective:** Establish a reproducible data pipeline using DVC.

**Deliverables:**
- DVC setup with a local remote (`~/dvc_storage/ACIS-Insurance-Analytics`).
- Tracked datasets: `insurance_data.txt` (raw) and `insurance_data_cleaned.txt` (processed).
- `task-2` branch with 10+ commits.

**Features:**
- Versioned datasets for auditability and reproducibility.
- Automation of data cleaning and preprocessing steps.

---

### **Task 3: A/B Hypothesis Testing**
**Objective:** Validate risk drivers (provinces, zip codes, gender) using metrics such as **Claim Frequency**, **Claim Severity**, and **Margin**.

**Deliverables:**
- Metrics: Claim Frequency, Claim Severity, Margin.
- Segmentation: Gauteng vs. Western Cape, zip codes 2000 vs. 8000, Male vs. Female.
- Statistical Tests: Chi-squared tests for frequency, t-tests for severity and margin.
- Script: `src/hypothesis_testing.py` using `insurance_data_cleaned.txt`.
- Report: `docs/hypothesis_testing_report.md` with detailed findings.
- Branch: `task-3` with 5+ commits.

**Key Findings:**
- **Rejected H₀₁:** Gauteng has a 55% higher claim frequency, while the Western Cape has a 26% higher claim severity.
- **Failed to Reject H₀₂, H₀₃, H₀₄:** No significant differences observed for zip codes or gender, but monitor zip code 8000 for increasing severity and negative margins.

**Recommendations:**
- Increase premiums in Gauteng by 10-15% and in the Western Cape by 5-10%.
- Monitor zip code 8000 for potential claim severity spikes.
- Review pricing strategies to address negative margins.
- Maintain gender-neutral pricing due to lack of significant evidence.

---

## **Contributions and Collaboration**

This project uses **GitHub Actions** for continuous integration and deployment (CI/CD). Contributions should follow these guidelines:
1. Create a new branch for each task (e.g., `task-1`, `task-2`, `task-3`).
2. Ensure all changes are tested locally before pushing to the repository.
3. Submit pull requests with clear descriptions of the changes made.

---

## **Dependencies**

All required dependencies are listed in `requirements.txt`. To install them, run:
```bash
pip install -r requirements.txt
```

---

## **Future Work**

- **Task 4:** Develop predictive models to forecast claim likelihood and severity.
- **Task 5:** Integrate machine learning pipelines for automated risk assessment.
- **Task 6:** Explore additional data sources (e.g., socioeconomic data) to enhance risk profiling.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

- **Data Source:** Historical insurance claim data provided by AlphaCare Insurance Solutions.
- **Tools Used:** Python, DVC, GitHub Actions, Pandas, Matplotlib, SciPy.
- **Inspiration:** This project was inspired by real-world challenges in the insurance industry.

---
