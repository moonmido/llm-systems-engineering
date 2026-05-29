
from datasets import load_dataset

"""### **DATASET**"""

# Load our data
data = load_dataset("cornell-movie-review-data/rotten_tomatoes")
data

"""### **Text Classification with Representation Models**

***Using a Task-specific Model***
"""

from transformers import pipeline

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipe= pipeline(
    model=model_path,
    tokenizer=model_path,
    return_all_scores=True,
)

import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

y_pred = []

for output in tqdm(pipe(KeyDataset(data["test"], "text")),
                   total=len(data["test"])):

    assignment = 1 if output["label"].lower() == "positive" else 0
    y_pred.append(assignment)

from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

evaluate_performance(data["test"]["label"], y_pred)