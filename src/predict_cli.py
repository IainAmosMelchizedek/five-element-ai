# ---- src/predict_cli.py ----
import torch
import pandas as pd
from src.model import AcupunctureModel
from src.preprocess import preprocess_data


def get_user_input():
    """
    Prompt the user to enter TCM-related symptoms and conditions.

    Returns:
        dict: A dictionary containing user input values.
    """
    print("Please enter the following information:")
    symptom_1 = input("Symptom 1: ").strip()
    symptom_2 = input("Symptom 2: ").strip()
    symptom_3 = input("Symptom 3: ").strip()
    tongue_color = input("Tongue color: ").strip()
    pulse_type = input("Pulse type: ").strip()
    spiritual_goal = input("Spiritual goal: ").strip()

    return {
        "symptom_1": symptom_1,
        "symptom_2": symptom_2,
        "symptom_3": symptom_3,
        "tongue_color": tongue_color,
        "pulse_type": pulse_type,
        "spiritual_goal": spiritual_goal
    }


def prepare_sample_input(sample_input, reference_df):
    input_df = pd.DataFrame([sample_input])
    full_df = pd.concat([reference_df.drop('acupoint', axis=1), input_df], ignore_index=True)
    encoded_df = pd.get_dummies(full_df)
    encoded_sample = encoded_df.tail(1)
    return torch.tensor(encoded_sample.values, dtype=torch.float32)


def main():
    df = pd.read_csv('data/synthetic_acupuncture_data.csv')
    df_encoded, label_encoder = preprocess_data(df)

    input_dim = df_encoded.drop('acupoint', axis=1).shape[1]
    output_dim = len(label_encoder.classes_)

    model = AcupunctureModel(input_dim, output_dim)
    model.load_state_dict(torch.load('saved_models/acupuncture_model.pth'))
    model.eval()

    user_input = get_user_input()
    input_tensor = prepare_sample_input(user_input, df)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
        predicted_acupoint = label_encoder.inverse_transform([predicted_index])[0]

    print(f"\nPredicted Acupoint: {predicted_acupoint}")


if __name__ == '__main__':
    main()
