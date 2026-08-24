import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualize:
    def __init__(self, df):
        """
        Initialize the Visualizer with a pandas DataFrame.
        """
        self.df = df

    def univariate_analysis(self, num_cols):
        """
        Perform univariate analysis by plotting histograms and KDEs
        for specified numerical columns.
        """
        print("\n--- Univariate Analysis ---")
        for col in num_cols:
            if col in self.df.columns:
                plt.figure(figsize=(8, 4))
                sns.histplot(self.df[col].dropna(), bins=50, kde=True)
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.show()
            else:
                print(f"Column '{col}' not found in DataFrame.")

    def categorical_analysis(self, cat_cols):
        """
        Visualize the frequency counts of specified categorical columns.
        """
        print("\n--- Categorical Analysis ---")
        for col in cat_cols:
            if col in self.df.columns:
                plt.figure(figsize=(8, 4))
                sns.countplot(x=col, data=self.df, order=self.df[col].value_counts().index)
                plt.title(f"Count of {col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Column '{col}' not found in DataFrame.")

    def bivariate_analysis(self):
        """
        Analyze the average Loss Ratio per Province.
        Loss Ratio = TotalClaims / TotalPremium
        """
        print("\n--- Bivariate Analysis: Loss Ratio by Province ---")
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            loss_by_province = self.df.groupby('Province')['LossRatio'].mean().sort_values()

            plt.figure(figsize=(10, 5))
            loss_by_province.plot(kind='barh', color='skyblue')
            plt.title("Average Loss Ratio by Province")
            plt.xlabel("Loss Ratio")
            plt.ylabel("Province")
            plt.tight_layout()
            plt.show()
        else:
            print("Missing required columns: 'TotalClaims' and/or 'TotalPremium'.")

    def outlier_detection(self, num_cols):
        """
        Detect outliers using box plots for specified numerical columns.
        """
        print("\n--- Outlier Detection ---")
        for col in num_cols:
            if col in self.df.columns:
                plt.figure(figsize=(8, 4))
                sns.boxplot(x=self.df[col].dropna(), color='orange')
                plt.title(f"Outliers in {col}")
                plt.xlabel(col)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Column '{col}' not found in DataFrame.")

    def gender_comparison(self):
        """
        Compare Loss Ratio by Gender using box plots.
        """
        print("\n--- Gender Comparison of Loss Ratio ---")
        if 'LossRatio' not in self.df.columns:
            if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
                self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            else:
                print("Cannot compute Loss Ratio. Missing 'TotalClaims' and/or 'TotalPremium'.")
                return

        if 'Gender' in self.df.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x='Gender', y='LossRatio', data=self.df)
            plt.title("Loss Ratio by Gender")
            plt.tight_layout()
            plt.show()
        else:
            print("Column 'Gender' not found in DataFrame.")

    def temporal_analysis(self):
        """
        Analyze trends over time based on the 'TransactionMonth' column.
        Aggregates and plots total premiums and claims per month.
        """
        print("\n--- Temporal Analysis ---")
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(self.df['TransactionMonth'], errors='coerce')
            monthly_stats = self.df.groupby(self.df['TransactionMonth'].dt.to_period('M')).agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            })

            monthly_stats.index = monthly_stats.index.astype(str)  # Convert PeriodIndex to string for plotting
            monthly_stats.plot(kind='line', figsize=(10, 5), marker='o')
            plt.title("Monthly Total Premiums vs Total Claims")
            plt.ylabel("Amount")
            plt.xlabel("Month")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("Column 'TransactionMonth' not found in DataFrame.")

    def top_insights(self):
        """
        Display top 3 visual insights:
        1. Loss Ratio by Vehicle Type
        2. Claim Severity Distribution
        3. High Risk Vehicle Models
        """
        print("\n--- Top 3 Insightful Visualizations ---")

        # Ensure LossRatio is computed
        if 'LossRatio' not in self.df.columns:
            if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
                self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            else:
                print("Cannot compute Loss Ratio. Missing 'TotalClaims' and/or 'TotalPremium'.")
                return

        # 1. Loss Ratio by Vehicle Type
        if 'VehicleType' in self.df.columns:
            loss_by_vehicle = self.df.groupby('VehicleType')['LossRatio'].mean().sort_values(ascending=False).head(10)
            plt.figure(figsize=(10, 5))
            loss_by_vehicle.plot(kind='bar', color='purple')
            plt.title("Top 10 Vehicle Types by Loss Ratio")
            plt.ylabel("Loss Ratio")
            plt.xlabel("Vehicle Type")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("Column 'VehicleType' not found.")

        # 2. Claim Severity Distribution
        if 'TotalClaims' in self.df.columns:
            claims_only = self.df[self.df['TotalClaims'] > 0]
            plt.figure(figsize=(10, 5))
            sns.histplot(claims_only['TotalClaims'], bins=50, kde=True)
            plt.title("Distribution of Claim Amounts (for claimed policies only)")
            plt.xlabel("Total Claims")
            plt.tight_layout()
            plt.show()
        else:
            print("Column 'TotalClaims' not found.")

        # 3. High Risk Vehicle Models
        if 'Model' in self.df.columns:
            high_risk_models = self.df.groupby('Model')['LossRatio'].mean().sort_values(ascending=False).head(10)
            plt.figure(figsize=(10, 5))
            high_risk_models.plot(kind='bar', color='red')
            plt.title("Top 10 Vehicle Models by Loss Ratio")
            plt.ylabel("Loss Ratio")
            plt.xlabel("Model")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("Column 'Model' not found.")
