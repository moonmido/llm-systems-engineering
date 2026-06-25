"""

User Request

      ↓
Orchestrator

      ↓
Generate Tasks

      ↓
      
Send()

      ↓

Workers (Parallel)

      ↓
Generate Code

      ↓
Synthesizer

      ↓
Final Output
"""

! pip install langchain_core langchain-nvidia-ai-endpoints langgraph

from typing import Annotated, List
import operator
from langgraph.types import Send
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

llm = ChatNVIDIA(
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv('API_KEY'),
    temperature=0
)

class State(TypedDict):
    project: str
    tasks: list[str]
    generated_code: Annotated[list[str], operator.add]
    final_output: str

class WorkerState(TypedDict):
    task: str
    generated_code: Annotated[list[str], operator.add]

def orchestrator(state: State):
    """
    Break project into tasks
    """

    project = state["project"]

    prompt = f"""
    You are a software architect.

    Break this project into coding tasks.

    Project:
    {project}

    Return only a Python list.

    Example:
    ["Entity","Repository","Service","Controller"]
    """

    response = llm.invoke(prompt)

    try:
        tasks = eval(response.content)
    except:
        tasks = [
            "Entity",
            "Repository",
            "Service",
            "Controller"
        ]

    return {
        "tasks": tasks
    }

def assign_workers(state: State):

    return [
        Send(
            "code_generator",
            {
                "task": task
            }
        )
        for task in state["tasks"]
    ]

def code_generator(state: WorkerState):

    task = state["task"]

    prompt = f"""
    Generate Spring Boot code for:

    {task}

    Return only code.
    """

    response = llm.invoke(prompt)

    return {
        "generated_code": [
            f"""
====================
{task}
====================

{response.content}
"""
        ]
    }

def synthesizer(state: State):

    final_result = "\n\n".join(
        state["generated_code"]
    )

    return {
        "final_output": final_result
    }

builder = StateGraph(State)

builder.add_node(
    "orchestrator",
    orchestrator
)

builder.add_node(
    "code_generator",
    code_generator
)

builder.add_node(
    "synthesizer",
    synthesizer
)

builder.add_edge(
    START,
    "orchestrator"
)

builder.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["code_generator"]
)

builder.add_edge(
    "code_generator",
    "synthesizer"
)

builder.add_edge(
    "synthesizer",
    END
)

graph = builder.compile()

result = graph.invoke(
    {
        "project": "Build a Notes CRUD API"
    }
)

print(result["final_output"])