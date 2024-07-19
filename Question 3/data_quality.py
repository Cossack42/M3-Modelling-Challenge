import pandas as pd

# Load the dataset (make sure 'your_data.csv' is the path to your CSV file)
df = pd.read_csv('dataset.csv')

# Data Cleaning
# Handle missing values
df.dropna(inplace=True)  # or use df.fillna(method) based on your data

# Remove outliers (e.g., based on the Z-score or IQR)
from scipy import stats
df = df[(np.abs(stats.zscore(df['TotalHomeless'])) < 3)]  # Example using Z-score