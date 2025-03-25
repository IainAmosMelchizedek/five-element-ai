# ---- src/predict.py ----
import torch
import pandas as pd
from src.model import AcupunctureModel
from src.preprocess import preprocess_data

import numpy as np

# Sample input: replace these with real user input or connect to an interface later
sample_input = {
    "symptom_1": "palpitations",
    "symptom_2": "restlessness",
    "symptom_3": "anxiety",
    "tongue_color": "red tip",
    "pulse_type": "rapid",
    "spiritual_goal": "breathe freely"  # representing desire for freedom
}




def prepare_sample_input(sample_input, reference_df):
    """
    One-hot encode a single sample input based on the training feature schema.

    Args:
        sample_input (dict): Dictionary containing the sample TCM features.
        reference_df (pd.DataFrame): The original training dataframe used for structure.

    Returns:
        torch.Tensor: One-hot encoded input tensor.
    """
    input_df = pd.DataFrame([sample_input])
    full_df = pd.concat([reference_df.drop('acupoint', axis=1), input_df], ignore_index=True)
    encoded_df = pd.get_dummies(full_df)
    encoded_sample = encoded_df.tail(1)
    return torch.tensor(encoded_sample.values, dtype=torch.float32)


def main():
    # Load original training data to get encoding structure
    df = pd.read_csv('data/synthetic_acupuncture_data.csv')
    df_encoded, label_encoder = preprocess_data(df)

    # Prepare model
    input_dim = df_encoded.drop('acupoint', axis=1).shape[1]
    output_dim = len(label_encoder.classes_)
    model = AcupunctureModel(input_dim, output_dim)
    model.load_state_dict(torch.load('saved_models/acupuncture_model.pth'))
    model.eval()

    # Prepare the sample input
    input_tensor = prepare_sample_input(sample_input, df)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_acupoint = label_encoder.inverse_transform([predicted_index])[0]

    print(f"Predicted Acupoint: {predicted_acupoint}")


if __name__ == '__main__':
    main()