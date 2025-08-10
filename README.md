# **AI Fine-Tuning Project — BERT Fine-Tune**

![Project Screenshot](https://github.com/Slimsnapz/Fine-Tunning-Bert-Model/blob/c5a3942de0506e8f8a2a9bd5fa987aa19ed09df0/distribution_of_data_labels.png)  
![Project Screenshot](https://github.com/Slimsnapz/Fine-Tunning-Bert-Model/blob/f7d85484c66717f49af1ca410b988e586b6563fc/frequency_of_data_labels.png)  
![Project Screenshot](https://github.com/Slimsnapz/Fine-Tunning-Bert-Model/blob/3da4d2e13afc4de95bd921c732b2c3e4decc5005/screenshots/confusion_matrix.png)  



## **Project Overview**

**Project Title:** AI Fine-Tuning — BERT Fine-Tune (Udemy Course Project)  
**Purpose:** Fine-tuned a pretrained BERT model on a downstream NLP task to demonstrate practical skills in transfer learning, model evaluation, and experiment tracking. This project was completed as part of the AI Fine-Tuning course on **Udemy** and is designed to showcase end-to-end fine-tuning workflows for potential employers.

**Objective**  
Show how to take a pretrained transformer (BERT), adapt it to a supervised task (classification / sequence labeling / regression as applicable), and produce reproducible results, clear evaluation, and simple inference code that can be re-used or extended in production.



## **Why this project matters to employers**

Yes — this kind of project can definitely serve as a strong portfolio piece **if** it clearly demonstrates:
- practical machine learning skills (fine-tuning pretrained models),
- software engineering hygiene (reproducible code, requirements, scripts),
- experiment tracking & evaluation (metrics, confusion matrices, loss curves),
- understanding of deployment constraints (model size, latency, inference code),
- clear documentation so non-technical reviewers and hiring managers can understand the impact.

Employers value projects that are well-documented, reproducible, and show measurable impact (improved accuracy, business-relevant metric, or a working demo). See the **Suggested improvements** section below for ways to make this project stand out even more.


## **Key Components & Features**

- **Model:** Pretrained BERT (e.g., `bert-base-uncased`) fine-tuned for a target task.
- **Task Types (adaptable):** Text classification (binary / multi-class), sentiment analysis, intent detection, or sequence labeling.
- **Data preparation:** tokenization (WordPiece), padding/truncation, train/val/test split, label encoding.
- **Training loop:** PyTorch (or Hugging Face `transformers`) training loop with AdamW, learning rate scheduler, gradient clipping, checkpointing.
- **Evaluation:** Accuracy, Precision, Recall, F1-score, confusion matrix, and training/validation loss curves.
- **Inference:** Lightweight inference script/notebook to run predictions on new examples.
- **Reproducibility:** fixed seeds, required packages (`requirements.txt`), and notebook documenting hyperparameters and results.
- **Artifacts:** saved model checkpoints, training logs, evaluation reports, and a short demo notebook.


## **Repository Contents**

```
/ (root)
├─ notebooks/
│  └─ FineTunningBert.ipynb          # Core notebook with training and evaluation (Udemy project)
├─ src/
│  ├─ data_loader.py                 # data loading & preprocessing
│  ├─ train.py                       # training script (CLI)
│  ├─ eval.py                        # evaluation script
│  └─ predict.py                     # simple inference script
├─ models/
│  └─ checkpoints/                   # saved model checkpoints (optional, large files excluded)
├─ requirements.txt                  # pip install -r requirements.txt
├─ README.md                         # (this file)
└─ screenshots/
   └─ training_logs.png              # training loss / eval metrics screenshot
```



## **How to run (quickstart)**
```

2. **Create a virtual environment & install requirements**
```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

3. **Prepare data**
- Place your dataset files in `data/` (example: `data/train.csv`, `data/val.csv`, `data/test.csv`).  
- Expected columns (example): `text`, `label`.  
- Update the path/column names in `src/data_loader.py` or the notebook.

4. **Train (using train.py)**
```bash
python src/train.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/train.csv \
  --valid_file data/val.csv \
  --output_dir models/checkpoints/exp1 \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5
```

5. **Evaluate**
```bash
python src/eval.py --model_dir models/checkpoints/exp1 --test_file data/test.csv
```

6. **Predict / Inference**
```bash
python src/predict.py --model_dir models/checkpoints/exp1 --input
```



## **Core Files & Notebooks**

- `notebooks/FineTunningBert.ipynb` — interactive notebook containing:
  - data inspection & cleaning,
  - tokenizer and dataset creation,
  - training loop (or Hugging Face Trainer) with metrics logging,
  - evaluation and confusion matrix plotting,
  - inference examples.

- `src/train.py` — production-ready training script (CLI) with configurable hyperparameters (epochs, lr, batch size, logging).

- `src/predict.py` — example inference script demonstrating tokenization and label decoding.



## **Modeling Details**

- **Backbone:** `bert-base-uncased` (Hugging Face Transformers)
- **Optimizer:** AdamW
- **Scheduler:** linear warmup + decay
- **Loss:** CrossEntropyLoss (classification)
- **Regularization:** weight decay, gradient clipping
- **Batch size / epochs:** tuned per dataset — sample: batch_size=16, epochs=2
- **Evaluation metrics:** Accuracy / Precision / Recall / F1 (macro & micro), confusion matrix



## **Results**

> Replace the example values below with the real results from your notebook.

- Best validation accuracy: **0.90**  
- Best validation F1 (macro): **0.86**  
- Test accuracy: **0.90**  
- Test F1: **0.88**









