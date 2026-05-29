
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

"""### **Classification Tasks that Leverage Embeddings**

***Supervised Classification***
"""

from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)

train_embeddings.shape

from sklearn.linear_model import LogisticRegression

# Train a Logistic Regression on our train embeddings
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

# Predict previously unseen instances
y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)

""" **Without Classifier ? Let's see !**"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Average the embeddings of all documents in each target label
df = pd.DataFrame(np.hstack([train_embeddings, np.array(data["train"]["label"]).reshape(-1, 1)]))
averaged_target_embeddings = df.groupby(768).mean().values

# Find the best matching embeddings between evaluation documents and target embeddings
sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

# Evaluate the model
evaluate_performance(data["test"]["label"], y_pred)

"""**Zero-shot Classification**"""

# Create embeddings for our labels
label_embeddings = model.encode(["A negative review",  "A positive review"])

from sklearn.metrics.pairwise import cosine_similarity

# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

evaluate_performance(data["test"]["label"], y_pred)

"""**What would happen if you were to use different descriptions?**"""

# Create embeddings for our labels
label_embeddings = model.encode(["A very negative movie review",  "A very positive movie review"])

from sklearn.metrics.pairwise import cosine_similarity

# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

evaluate_performance(data["test"]["label"], y_pred)