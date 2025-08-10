"""
predict.py
Load a fine-tuned model and run prediction on input texts.
Usage:
python predict.py --model_dir models/checkpoints/exp1 --text "I love this!"
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_dir: str, device: str = None):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def get_prediction(text: str, tokenizer, model, device, id2label=None):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=1).item()
    if id2label is not None:
        return id2label.get(pred, str(pred))
    return str(pred)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--file", type=str, default=None, help="Path to a text file with one example per line")
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer, model, device = load_model(args.model_dir)
    # Attempt to load id2label mapping from the model config if available
    id2label = None
    try:
        config = model.config
        if hasattr(config, "id2label"):
            id2label = config.id2label
    except Exception:
        id2label = None

    if args.text:
        print(get_prediction(args.text, tokenizer, model, device, id2label=id2label))
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                print(text, "->", get_prediction(text, tokenizer, model, device, id2label=id2label))
    else:
        print("Provide --text or --file for predictions")

if __name__ == "__main__":
    # write this script file to disk so you can save it as a .py file (as requested)
    content = open(__file__, "r", encoding="utf-8").read()
    with open("predict.py", "w", encoding="utf-8") as f:
        f.write(content)
    main()
