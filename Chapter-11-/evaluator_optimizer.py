

! pip install langchain_core langchain-openai langgraph langchain-nvidia-ai-endpoints

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from typing_extensions import Literal
from typing import Annotated, List

class State(TypedDict):
  topic:str
  generated_prompt:str
  feedback:str
  great_or_not:str

class Feedback(BaseModel):
    grade: Literal["GOOD", "NOT GOOD"] = Field(
        description="Decide if the prompt is great or not.",
    )
    feedback: str = Field(
        description="If the prompt is not great, provide feedback on how to improve it.",
    )

from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

llm = ChatNVIDIA(model=os.getenv("LLM_MODEL"),api_key=os.getenv("LLM_API_KEY"))

evaluator = llm.with_structured_output(Feedback)

def llm_call_generator(state: State):
    """LLM generates a prompt"""

    if state.get("feedback"):
        msg = llm.invoke(
            f"Write a prompt about {state['topic']} but take into account the feedback: {state['feedback']}"
        )
    else:
        msg = llm.invoke(f"Write a prompt about {state['topic']}")
    return {"generated_prompt": msg.content}

def llm_evaluator(state:State):
  """LLM evaluate a prompt"""

  msg = evaluator.invoke(f"evaluate {state['generated_prompt']}")
  if msg is None:
      # Handle the case where the evaluator returns None,
      # perhaps due to an API error or inability to generate structured output.
      # For now, we'll return a default 'NOT GOOD' feedback.
      print("Warning: Evaluator returned None. Returning default 'NOT GOOD' feedback.")
      return {"great_or_not": "NOT GOOD", "feedback": "Evaluator failed to provide feedback. Check API key or model availability."}
  return {"great_or_not": msg.grade, "feedback": msg.feedback}

def route_prompt(state: State):
    """Route back to prompt generator or end based upon feedback from the evaluator"""

    if state["great_or_not"] == "GOOD":
        return "Accepted"
    elif state["great_or_not"] == "NOT GOOD":
        return "Rejected + Feedback"

optimizer_builder = StateGraph(State)

# Add the nodes
optimizer_builder.add_node("llm_call_generator", llm_call_generator)
optimizer_builder.add_node("llm_evaluator", llm_evaluator)

# Add edges to connect nodes
optimizer_builder.add_edge(START, "llm_call_generator")
optimizer_builder.add_edge("llm_call_generator", "llm_evaluator")
optimizer_builder.add_conditional_edges(
    "llm_evaluator",
    route_prompt,
    {
        "Accepted": END,
        "Rejected + Feedback": "llm_call_generator",
    },
)

# Compile the workflow
optimizer_workflow = optimizer_builder.compile()

# Show the workflow
display(Image(optimizer_workflow.get_graph().draw_mermaid_png()))

# Invoke
state = optimizer_workflow.invoke({"topic": "Write image prompt"})
print(state["generated_prompt"])