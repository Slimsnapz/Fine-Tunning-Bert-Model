"""
train.py
Training script for fine-tuning a Hugging Face BERT model for classification.
Usage example:
python train.py --model_name_or_path bert-base-uncased --train_file data/train.csv --valid_file data/val.csv --output_dir models/checkpoints/exp1 --epochs 3 --batch_size 16 --lr 2e-5
"""

import argparse
import os
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score

# Simple compute_metrics (from notebook)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    acc = accuracy_score(y_true=labels, y_pred=preds)
    return {'accuracy': acc, 'f1': f1}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets from CSV
    data_files = {"train": args.train_file}
    if args.valid_file:
        data_files["validation"] = args.valid_file
    dataset = load_dataset("csv", data_files=data_files)

    # Auto-detect text & label columns (common names)
    # Expecting 'text' and 'label' columns; adjust if necessary.
    # If label column is string, map to ints:
    if dataset['train'].features.get("label") is None and "label" not in dataset['train'].column_names:
        # try common alternatives
        possible_label_cols = [c for c in dataset['train'].column_names if "label" in c.lower() or "target" in c.lower()]
        if possible_label_cols:
            label_col = possible_label_cols[0]
            dataset = dataset.rename_column(label_col, "label")
        else:
            raise ValueError("Could not find a label column in the training data. Please ensure a 'label' column exists.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Tokenize
    def tokenize_fn(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=args.max_length)
    dataset = dataset.map(tokenize_fn, batched=True)

    # Set format for PyTorch
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Load model
    # Determine num_labels automatically
    labels = np.unique(dataset['train']['label'])
    num_labels = int(len(labels))
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch" if "validation" in dataset else "no",
        save_strategy="epoch",
        learning_rate=args.lr,
        load_best_model_at_end=True if "validation" in dataset else False,
        metric_for_best_model="f1",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset.get('train', None),
        eval_dataset=dataset.get('validation', None),
        compute_metrics=compute_metrics if "validation" in dataset else None,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    # write this script file to disk so you can save it as a .py file (as requested)
    content = open(__file__, "r", encoding="utf-8").read()
    with open("train.py", "w", encoding="utf-8") as f:
        f.write(content)
    main()
