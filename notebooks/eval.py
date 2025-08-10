"""
eval.py
Evaluate a saved model on a test CSV file.
Usage:
python eval.py --model_dir models/checkpoints/exp1 --test_file data/test.csv
"""
import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {'accuracy': acc, 'f1': f1}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # Load test dataset
    dataset = load_dataset("csv", data_files={"test": args.test_file})['test']

    # Tokenize
    def tokenize_fn(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=args.max_length)
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    trainer = Trainer(model=model, tokenizer=tokenizer, compute_metrics=compute_metrics)
    results = trainer.predict(dataset)
    print("Evaluation results:")
    print(results.metrics)

if __name__ == "__main__":
    # write this script file to disk so you can save it as a .py file (as requested)
    content = open(__file__, "r", encoding="utf-8").read()
    with open("eval.py", "w", encoding="utf-8") as f:
        f.write(content)
    main()
