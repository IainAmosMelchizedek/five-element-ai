# Five Element AI

An AI-powered assistant trained on Traditional Chinese Medicine (TCM) principles to recommend acupuncture points based on symptoms, tongue diagnosis, pulse quality, and spiritual goals.

Five Element AI is a neural network trained on Traditional Chinese Medicine (TCM) principles to recommend personalized acupuncture points based on physical symptoms, pulse and tongue diagnoses, and spiritual goals.


---

## Created by
**Iain Amos Melchizedek**  
[GitHub Profile](https://github.com/IainAmosMelchizedek)

---

## What It Does

This project simulates a diagnostic assistant grounded in the Five Element theory of TCM. It learns patterns from synthetic acupuncture data and predicts the optimal acupoint for treatment.

### Example Input:
- Symptom 1: poor appetite  
- Symptom 2: bloating  
- Symptom 3: fatigue  
- Tongue color: pale  
- Pulse type: slippery  
- Spiritual goal: feel nurtured  

### Predicted Output:
`Predicted Acupoint: ST36`

---

## Tools & Libraries

- **PyTorch** for neural networks
- **scikit-learn** for encoding
- **pandas** & **numpy** for data
- **matplotlib** (optional for visuals)
- Fully modular and CLI-ready

---

## How to Use

### Clone & Setup
```bash
git clone https://github.com/IainAmosMelchizedek/five-element-ai.git
cd five-element-ai

# Optional: Create and activate virtual environment
conda create -n five_element_ai python=3.10
conda activate five_element_ai

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

#### Step 1: Generate Data
```bash
python src/generate_data.py
```

#### Step 2: Preprocess Data
```bash
python src/preprocess.py
```

#### Step 3: Train Model
```bash
python src/train.py
```

#### Step 4: Evaluate Performance
```bash
python src/evaluate.py
```

#### Step 5: Make Predictions (example-based)
```bash
python src/predict.py
```

#### Optional: Interactive CLI
```bash
python src/predict_cli.py
```

---

## Project Structure

```bash
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
```

---

## Why This Matters

This project isn’t just about AI — it’s about integrating **spirituality**, **holistic healing**, and **data science** into something that can help people reconnect with themselves. It is a step toward **humanitarian AI**, where machine learning serves meaning.

> _"Integrity First, Future Forward."_

---

## License

This project is open to the world — for education, inspiration, and healing.

> I’m offering this to the world, with no ego — but please honor the spirit it came from: Melchizedek  
> **Attribution appreciated.**

Use it, build with it, learn from it. Just remember the heart behind it.

---

## Contact
For collaboration or consulting, please reach out via GitHub or LinkedIn.




