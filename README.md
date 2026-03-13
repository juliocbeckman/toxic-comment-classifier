# Toxic Comment Classifier by Julio Beckman

A binary toxicity classifier for social media comments, comparing three progressively complex approaches:

1. **Baseline**: TF-IDF (unigrams) + Logistic Regression
2. **Improved**: TF-IDF (uni+bigrams) + Linear SVM
3. **DistilBERT**: Fine-tuned transformer with weighted cross-entropy loss

Trained on the [Jigsaw Toxic Comment Classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and [Jigsaw Unintended Bias](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) datasets.

**Key techniques:** random oversampling, class-weighted loss, and validation-set threshold tuning to address the ~90/10 class imbalance in the dataset.

## Demo

Type any comment and see how all three models classify it in real time.

```bash
git clone https://github.com/YOUR_USERNAME/toxic-comment-classifier.git
cd toxic-comment-classifier
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:7860` in your browser.

## Project Structure

```
toxic-comment-classifier/
├── app.py                  # Gradio demo application
├── requirements.txt        # Python dependencies
├── README.md
├── .gitignore
├── notebooks/
│   └── training.ipynb      # Full training pipeline with outputs
├── models/
│   ├── baseline_clf.joblib
│   ├── improved_clf_1.joblib
│   ├── config.json
│   └── distilbert-toxic/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files
└── assets/
    └── demo_screenshot.png
```

## Models

| Model | Approach | Accuracy | Precision (toxic) | Recall (toxic) | F1 (toxic) |
|-------|----------|----------|-------------------|----------------|------------|
| Baseline | TF-IDF unigrams + Logistic Regression | 93.3% | 0.633 | 0.823 | 0.716 |
| Improved | TF-IDF uni+bigrams + Linear SVM | 95.3% | 0.807 | 0.707 | 0.754 |
| **DistilBERT** | **Fine-tuned transformer + weighted loss** | **94.9%** | **0.700** | **0.867** | **0.774** |

DistilBERT achieves the highest F1 and recall, catching 86.7% of toxic comments with the fewest false negatives (324 out of 2,434 toxic test comments). For content moderation, where letting toxic comments through is more costly than over-flagging, this makes it the strongest choice.

### Class Imbalance Handling

The Jigsaw dataset is approximately 90% non-toxic / 10% toxic. Three techniques were applied:

1. **Random oversampling** of the minority class — training set balanced to 200,341 examples per class (400,682 total)
2. **Weighted cross-entropy loss** for DistilBERT via a custom `WeightedTrainer` subclass
3. **Threshold tuning** on the validation set — baseline optimal threshold shifted from 0.50 to 0.75, DistilBERT to 0.90

## Known Limitations

### 1. Implicit Hate Speech and Conspiracy Tropes

The model fails to flag statements like *"jews control the world"* or *"I just think people like you shouldn't be allowed to vote."* These express exclusionary or hateful ideologies using civil language without profanity. The Jigsaw dataset is heavily skewed toward explicit toxicity, so the model has almost no training signal for politely-worded hate.

**Production approach:** Large-scale annotation campaigns targeting implicit hate, retrieval-augmented detection using curated knowledge bases (ADL, SPLC), and larger transformer models with stronger contextual understanding.

### 2. Word Sense Disambiguation

The model incorrectly flags *"what is the difference between a donkey and an ass"* as toxic. It cannot distinguish between "ass" as profanity and "ass" as a synonym for donkey.

**Production approach:** Larger language models with broader pre-training data handle polysemy better. Contextual rule layers that check surrounding words before flagging could reduce false positives.

### 3. Victim vs. Aggressor Distinction

Comments like *"They keep calling me stupid and it's really hurting me"* are flagged as toxic even though the speaker is reporting harassment, not perpetrating it.

**Production approach:** Multi-turn context windows or labeling schemes that distinguish between toxic speech and reported toxic speech. Human-in-the-loop review for borderline cases.

### 4. Multilingual and Code-Switched Content

Mixed-language comments like *"tu opinión es una basura and you know it"* are missed entirely. The model was trained on English text only.

**Production approach:** Multilingual transformers (XLM-RoBERTa) or language-detection preprocessing with language-specific classifiers.

### 5. Implicit Threats

Comments like *"Keep it up and I will find you and make you regret it"* and *"I hope your whole family suffers"* are missed. Without explicit violent language, the model treats them as safe.

**Production approach:** Dedicated threat-detection models trained on threat-specific datasets.

### Summary

These limitations highlight a core reality of content moderation: no single classifier is sufficient. Production platforms use ensemble systems combining specialized models, retrieval systems, human review queues, and user reporting mechanisms. This project demonstrates a strong baseline pipeline with industry-standard imbalance-handling techniques, but a production deployment would require the additional layers described above.

## Training

The full training pipeline is documented in [`notebooks/training.ipynb`](notebooks/training.ipynb). It was developed in Google Colab with a T4 GPU. Full training takes approximately 10 minutes for DistilBERT (2 epochs on 80k examples).

### Data

This project uses two Kaggle datasets (not included in this repo due to size):

1. [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) — `train.csv` (~160k comments)
2. [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) — `train.csv` (~1.8M comments, 100k sampled)

## License

This project is for educational purposes. The Jigsaw datasets are released under CC0.
