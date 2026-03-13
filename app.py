"""
Toxic Comment Classifier — Gradio Demo
========================================
Interactive demo for comparing three toxicity classification models:
  1. Baseline: TF-IDF (unigrams) + Logistic Regression
  2. Improved: TF-IDF (uni+bigrams) + Linear SVM
  3. DistilBERT fine-tuned with weighted cross-entropy loss

Usage:
  pip install -r requirements.txt
  python app.py
"""

import gradio as gr
import numpy as np
import torch
import joblib
import json
import re
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── config ────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── load models ───────────────────────────────────────────────────────
print("loading models...")

baseline_clf = joblib.load(os.path.join(MODEL_DIR, "baseline_clf.joblib"))
improved_clf_1 = joblib.load(os.path.join(MODEL_DIR, "improved_clf_1.joblib"))

model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(MODEL_DIR, "distilbert-toxic")
)
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_DIR, "distilbert-toxic")
)

with open(os.path.join(MODEL_DIR, "config.json"), "r") as f:
    config = json.load(f)

best_thresh_baseline = config["best_thresh_baseline"]
best_thresh_bert = config["best_thresh_bert"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"models loaded — device: {device}")
print(f"thresholds — baseline: {best_thresh_baseline}, DistilBERT: {best_thresh_bert}")

# ── text cleaning (must match training pipeline) ─────────────────────
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
USER_PATTERN = re.compile(r"@\w+")
NEWLINE_PATTERN = re.compile(r"\n+")


def clean_comment(text: str) -> str:
    """normalize a raw comment for model input."""
    text = text.lower()
    text = URL_PATTERN.sub(" URL ", text)
    text = USER_PATTERN.sub(" USER ", text)
    text = NEWLINE_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── prediction logic ─────────────────────────────────────────────────
def classify_comment(text: str):
    """run all three models on a single comment and return formatted results."""

    if not text or not text.strip():
        return "Please enter a comment to classify.", "", ""

    cleaned = clean_comment(text)

    # baseline — logistic regression with tuned threshold
    base_proba = baseline_clf.predict_proba([cleaned])[0][1]
    base_pred = int(base_proba >= best_thresh_baseline)

    # improved — LinearSVC decision function
    imp1_pred = improved_clf_1.predict([cleaned])[0]
    imp1_score = improved_clf_1.decision_function([cleaned])[0]

    # DistilBERT — softmax probabilities with tuned threshold
    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]
    bert_pred = int(probs[1] >= best_thresh_bert)
    bert_proba = probs[1]

    # format results
    label = {0: "KEEP  ✅  non-toxic", 1: "DELETE  🚫  toxic"}

    baseline_result = (
        f"**{label[base_pred]}**\n\n"
        f"P(toxic) = {base_proba:.3f} · threshold = {best_thresh_baseline:.2f}"
    )

    svm_result = (
        f"**{label[imp1_pred]}**\n\n"
        f"decision score = {imp1_score:.3f}"
    )

    bert_result = (
        f"**{label[bert_pred]}**\n\n"
        f"P(toxic) = {bert_proba:.3f} · threshold = {best_thresh_bert:.2f}"
    )

    return baseline_result, svm_result, bert_result


# ── example inputs ───────────────────────────────────────────────────
examples = [
    ["Great article, thanks for sharing this!"],
    ["You are the dumbest person I have ever met."],
    ["I disagree with your point but I see where you are coming from."],
    ["Shut up you piece of garbage."],
    ["They keep calling me stupid in the group chat and it's really hurting me."],
    ["I just think people like you shouldn't be allowed to vote, that's all."],
    ["What a killer performance by the band last night!"],
    ["I love how you always manage to make everything worse."],
]

# ── gradio interface ─────────────────────────────────────────────────
with gr.Blocks(
    title="Toxic Comment Classifier",
    theme=gr.themes.Soft(
        primary_hue="red",
        secondary_hue="gray",
        neutral_hue="slate",
    ),
) as demo:

    gr.Markdown(
        """
        ## Toxic Comment Classifier
        Type any comment below to see how three different models classify it.
        Each model uses a different approach to detect toxic content.

        | Model | Approach | F1 (toxic) |
        |-------|----------|-----------|
        | **Baseline** | TF-IDF unigrams + Logistic Regression | 0.716 |
        | **Improved** | TF-IDF uni+bigrams + Linear SVM | 0.754 |
        | **DistilBERT** | Fine-tuned transformer + weighted loss | 0.774 |
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Enter a comment",
            placeholder="Type a comment to analyze...",
            lines=3,
            scale=4,
        )
        submit_btn = gr.Button("Classify", variant="primary", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline (TF-IDF + LR)")
            baseline_output = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Improved (TF-IDF + SVM)")
            svm_output = gr.Markdown()
        with gr.Column():
            gr.Markdown("### DistilBERT")
            bert_output = gr.Markdown()

    gr.Examples(
        examples=examples,
        inputs=text_input,
        outputs=[baseline_output, svm_output, bert_output],
        fn=classify_comment,
        cache_examples=False,
    )

    gr.Markdown(
        """
        ---
        *Built by Julio Beckman · Jigsaw Toxic Comment Dataset
        · See README for failure cases*
        """
    )

    submit_btn.click(
        fn=classify_comment,
        inputs=text_input,
        outputs=[baseline_output, svm_output, bert_output],
    )

    text_input.submit(
        fn=classify_comment,
        inputs=text_input,
        outputs=[baseline_output, svm_output, bert_output],
    )

# ── launch ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()
