
python -m ragas_examples.rag_eval.rag

import pandas as pd

samples = [
    {"query": "What is Ragas 0.3?", "grading_notes": "- Ragas 0.3 is a library for evaluating LLM applications."},
    {"query": "How to install Ragas?", "grading_notes": "- install from source  - install from pip using ragas[examples]"},
    {"query": "What are the main features of Ragas?", "grading_notes": "organised around - experiments - datasets - metrics."}
]
pd.DataFrame(samples).to_csv("datasets/test_dataset.csv", index=False)

from ragas.metrics import DiscreteMetric
my_metric = DiscreteMetric(
    name="correctness",
    prompt = "Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],
)

@experiment()
async def run_experiment(row):
    response = rag_client.query(row["query"])

    score = my_metric.score(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"]
    )

    experiment_view = {
        **row,
        "response": response.get("answer", ""),
        "score": score.value,
        "log_file": response.get("logs", " "),
    }
    return experiment_view

"""# **Now whenever you make a change to your RAG pipeline, you can run the experiment and see how it affects the performance of your RAG.**

**You can now inspect the results by opening the experiments/experiment_name.csv file.**
"""