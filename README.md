


# Five Element AI

An AI-powered assistant trained on Traditional Chinese Medicine (TCM) principles to recommend acupuncture points based on symptoms, tongue diagnosis, pulse quality, and spiritual goals.

Five Element AI is a neural network trained on Traditional Chinese Medicine (TCM) principles to recommend personalized acupuncture points based on physical symptoms, pulse and tongue diagnoses, and spiritual goals.


## Created by
**Iain Amos Melchizedek**  
[GitHub Profile](https://github.com/IainAmosMelchizedek)

---

## What It Does

This project simulates a diagnostic assistant grounded in the Five Element theory of TCM. It learns patterns from synthetic acupuncture data and predicts the optimal acupoint for treatment.

### Example:
**Input:**
- Symptom 1: poor appetite
- Symptom 2: bloating
- Symptom 3: fatigue
- Tongue color: pale
- Pulse type: slippery
- Spiritual goal: feel nurtured

**Output:**
`Predicted Acupoint: ST36`

---

## Tools & Libraries

- `PyTorch` for neural networks
- `scikit-learn` for encoding
- `pandas` & `numpy` for data
- `matplotlib` for visuals (optional)
- Fully modular and CLI-ready

---

## How to Use

```bash
# Clone the repo
git clone https://github.com/IainAmosMelchizedek/five-element-ai.git
cd five-element-ai

# Create virtual environment (optional)
conda create -n five_element_ai python=3.10
conda activate five_element_ai

# Install dependencies
pip install -r requirements.txt

## Run Modules

# Step 1: Generate data
python src/generate_data.py

# Step 2: Preprocess data
python src/preprocess.py

# Step 3: Train model
python src/train.py

# Step 4: Evaluate performance
python src/evaluate.py

# Step 5: Make predictions
python src/predict.py

## Run with User Input (Interactive)

python src/predict_cli.py

########## PROJECT STRUCTURE 

five-element-ai/
├── data/
│   └── synthetic_acupuncture_data.csv
├── saved_models/
│   └── acupuncture_model.pth
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── predict_cli.py
├── requirements.txt
└── README.md

Why This Matters
This project isn’t just about AI — it’s about integrating spirituality, holistic healing, and data science into something that can help people reconnect with themselves. This is the beginning of humanitarian AI — where machine learning meets meaning.

"Integrity First, Future Forward."

License
Open for educational and spiritual research purposes. Attribution appreciated.

Contact:

For collaboration or consulting, please reach out via GitHub or LinkedIn.


---



