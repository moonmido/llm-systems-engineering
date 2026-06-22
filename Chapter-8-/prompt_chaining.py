

!pip install langgraph typing_extensions IPython

!pip install -qU langchain-nvidia-ai-endpoints

from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

llm = ChatNVIDIA(model="openai/gpt-oss-20b",api_key=os.getenv("LLM_API"))

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


class State(TypedDict):
   topic:str
   explain:str
   example:str
   quiz:str

def explain_topic(state: State):
    """Explain topic in simple way"""

    msg = llm.invoke(
        f"Explain {state['topic']} in a simple and beginner-friendly way"
    )
    return {"explain": msg.content}

def example_topic(state: State):
    """Give real-world example"""

    msg = llm.invoke(
        f"Give a real-world example of {state['topic']} and explain how it is used in practice"
    )
    return {"example": msg.content}

def quiz_topic(state: State):
    """Generate quiz questions"""

    msg = llm.invoke(
        f"Generate 3 quiz questions with answers about {state['topic']}. Keep it simple and educational."
    )
    return {"quiz": msg.content}

workflow = StateGraph(State)

#  Nodes
workflow.add_node("explain_topic", explain_topic)
workflow.add_node("example_topic", example_topic)
workflow.add_node("quiz_topic", quiz_topic)

#  Flow (sequential chaining)
workflow.add_edge(START, "explain_topic")
workflow.add_edge("explain_topic", "example_topic")
workflow.add_edge("example_topic", "quiz_topic")
workflow.add_edge("quiz_topic", END)

#  Compile graph
app = workflow.compile()

result = app.invoke({"topic": "JWT authentication"})
print(result["explain"])
print(result["example"])
print(result["quiz"])