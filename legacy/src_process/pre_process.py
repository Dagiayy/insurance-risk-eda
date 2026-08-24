import pandas as pd

class Pre_processor:
    def __init__(self, file_path):
        """
        Initialize the Preprocessor with the given file path.
        """
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Load the dataset from the specified file path using a pipe '|' delimiter.
        """
        try:
            self.df = pd.read_csv(self.file_path, sep="|", engine="python")
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def initial_exploration(self):
        """
        Display basic information about the dataset including shape, column names, and first few rows.
        """
        if self.df is not None:
            print("\n--- Initial Data Exploration ---")
            print(f"Shape of the dataset: {self.df.shape}")
            print("Column names:", self.df.columns.tolist())
            print("\nFirst 5 rows of the dataset:")
            print(self.df.head())
        else:
            print("Data not loaded. Please run `load_data()` first.")

    def data_quality_checks(self):
        """
        Perform data quality checks:
        - Check for missing values
        - Check data types
        - Remove duplicate rows
        """
        if self.df is not None:
            print("\n--- Data Quality Checks ---")

            # Check for missing values
            print("\nMissing values per column:")
            print(self.df.isnull().sum())

            # Display data types of each column
            print("\nData types:")
            print(self.df.dtypes)

            # Drop duplicate rows, if any
            initial_shape = self.df.shape
            self.df.drop_duplicates(inplace=True)
            final_shape = self.df.shape

            if initial_shape != final_shape:
                print(f"\nDropped {initial_shape[0] - final_shape[0]} duplicate rows.")
            else:
                print("\nNo duplicate rows found.")
        else:
            print("Data not loaded. Please run `load_data()` first.")

    def descriptive_stats(self):
        """
        Display descriptive statistics for key numerical columns.
        """
        if self.df is not None:
            print("\n--- Descriptive Statistics ---")
            num_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
            
            # Ensure selected columns exist in the DataFrame
            existing_cols = [col for col in num_cols if col in self.df.columns]
            if not existing_cols:
                print("No matching numerical columns found for statistics.")
                return

            print(self.df[existing_cols].describe())
        else:
            print("Data not loaded. Please run `load_data()` first.")
