# ---- src/preprocess.py ----
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data(path='data/synthetic_acupuncture_data.csv'):
    """
    Load the dataset from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    """
    Preprocess the data by one-hot encoding categorical features and
    label encoding the target variable.

    Args:
        df (pd.DataFrame): Raw DataFrame.

    Returns:
        Tuple[pd.DataFrame, LabelEncoder]: Encoded DataFrame and label encoder for acupoint.
    """
    df_encoded = pd.get_dummies(df, columns=['symptom_1', 'symptom_2', 'symptom_3', 'pulse_type', 'spiritual_goal', 'tongue_color'])
    label_encoder = LabelEncoder()
    df_encoded['acupoint'] = label_encoder.fit_transform(df['acupoint'])
    return df_encoded, label_encoder

def main():
    """CLI entry point for preprocessing."""
    df = load_data()
    df_encoded, label_encoder = preprocess_data(df)
    print("Preprocessing complete. Encoded feature shape:", df_encoded.shape)

if __name__ == '__main__':
    main()
