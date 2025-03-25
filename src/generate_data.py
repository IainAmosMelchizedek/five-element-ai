# ---- src/generate_data.py ----
import os
import random
import pandas as pd

# Dictionary for the Five Element model in Traditional Chinese Medicine (TCM)
five_elements = {
    "Wood": {
        "symptoms": ["anger", "red eyes", "tendon stiffness", "headache", "irritability"],
        "tongue_color": "red",
        "pulse_type": "wiry",
        "acupoint": "LV3",
        "spiritual_goals": ["gain clarity", "release anger", "embrace change"]
    },
    "Fire": {
        "symptoms": ["insomnia", "palpitations", "restlessness", "tongue ulcers", "anxiety"],
        "tongue_color": "red tip",
        "pulse_type": "rapid",
        "acupoint": "HT7",
        "spiritual_goals": ["feel joy", "open heart", "connect to purpose"]
    },
    "Earth": {
        "symptoms": ["fatigue", "poor appetite", "bloating", "heavy limbs", "worry"],
        "tongue_color": "pale",
        "pulse_type": "slippery",
        "acupoint": "ST36",
        "spiritual_goals": ["find stability", "feel nurtured", "process emotions"]
    },
    "Metal": {
        "symptoms": ["cough", "dry skin", "sadness", "nasal congestion", "grief"],
        "tongue_color": "white",
        "pulse_type": "weak",
        "acupoint": "LU9",
        "spiritual_goals": ["let go of grief", "breathe freely", "reclaim self-worth"]
    },
    "Water": {
        "symptoms": ["tinnitus", "cold limbs", "fear", "low back pain", "night sweating"],
        "tongue_color": "bluish",
        "pulse_type": "deep",
        "acupoint": "KI3",
        "spiritual_goals": ["face fears", "reconnect to source", "trust intuition"]
    }
}

def generate_data(num_samples=500):
    """
    Generate synthetic acupuncture data based on the Five Element TCM model.

    Args:
        num_samples (int): Number of synthetic samples to generate.

    Returns:
        pd.DataFrame: A DataFrame containing synthetic acupuncture data.
    """
    data = []
    for _ in range(num_samples):
        element = random.choice(list(five_elements.keys()))
        profile = five_elements[element]
        symptoms = random.sample(profile["symptoms"], k=3)
        data.append({
            "symptom_1": symptoms[0],
            "symptom_2": symptoms[1],
            "symptom_3": symptoms[2],
            "tongue_color": profile["tongue_color"],
            "pulse_type": profile["pulse_type"],
            "spiritual_goal": random.choice(profile["spiritual_goals"]),
            "acupoint": profile["acupoint"]
        })
    return pd.DataFrame(data)

def save_data(df, path='data/synthetic_acupuncture_data.csv'):
    """
    Save the generated DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): File path to save the CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")

def main():
    """Entry point for the CLI."""
    df = generate_data()
    save_data(df)

if __name__ == '__main__':
    main()
