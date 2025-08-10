"""
data_loader.py
Utility functions to load CSV data and prepare Hugging Face datasets and tokenization.
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from typing import Optional, Dict

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    Expecting at least columns: 'text' and 'label' (or adapt accordingly).
    """
    return pd.read_csv(path)

def df_to_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert a pandas DataFrame to a Hugging Face Dataset.
    """
    return Dataset.from_pandas(df.reset_index(drop=True))

def build_dataset_dict(train_df: pd.DataFrame = None,
                       valid_df: pd.DataFrame = None,
                       test_df: pd.DataFrame = None) -> DatasetDict:
    """
    Build a DatasetDict from pandas DataFrames (any of train/valid/test can be None).
    """
    data = {}
    if train_df is not None:
        data['train'] = df_to_dataset(train_df)
    if valid_df is not None:
        data['validation'] = df_to_dataset(valid_df)
    if test_df is not None:
        data['test'] = df_to_dataset(test_df)
    return DatasetDict(data)

def get_tokenize_function(tokenizer, text_column: str = "text"):
    """
    Returns a tokenize(batch) function that can be used with dataset.map(...)
    """
    def tokenize(batch):
        return tokenizer(batch[text_column], padding=True, truncation=True)
    return tokenize

# Optional: write this file to disk when executed (so the user can save it), as requested
if __name__ == "__main__":
    content = open(__file__, "r", encoding="utf-8").read()
    with open("data_loader.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved data_loader.py")
