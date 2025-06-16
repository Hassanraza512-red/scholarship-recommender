# app/data_utils.py

import pandas as pd
import os

def load_data(path='data/raw/scholarships_mock.csv'):
    """
    Load the scholarships dataset from the given path.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, encoding='latin1')

    expected_columns = {
        'Title', 'Provider', 'Description', 'Eligibility_Criteria',
        'Fields_of_Study', 'Amount', 'Deadline', 'Country', 'Application_Fee'
    }
    if not expected_columns.issubset(df.columns):
        raise ValueError(f"CSV is missing expected columns. Found: {df.columns.tolist()}")

    print(f"Loaded data with shape: {df.shape}")
    print(df.head())

    return df


