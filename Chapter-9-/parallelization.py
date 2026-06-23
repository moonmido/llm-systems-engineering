
! pip install -qU langchain-nvidia-ai-endpoints typing_extensions langgraph IPython

"""### **Student Report Generator**"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


class State(TypedDict):
    name: str
    grade: float
    evaluation: str
    motivation: str
    recommendation: str
    report: str

from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os

llm = ChatNVIDIA(model="openai/gpt-oss-20b",api_key=os.gentenv('LLM_API'))

def grade_evaluator(state: State):
    """Evaluate student's grade"""

    msg = llm.invoke(
        f"""
        Student Name: {state['name']}
        Grade: {state['grade']}/20

        Evaluate the student's performance in 2-3 sentences.
        """
    )

    return {"evaluation": msg.content}

def motivation_generator(state: State):
    """Generate motivational message"""

    msg = llm.invoke(
        f"""
        Student Name: {state['name']}
        Grade: {state['grade']}/20

        Write a short motivational message for this student.
        """
    )

    return {"motivation": msg.content}

def recommendation_generator(state: State):
    """Generate study recommendations"""

    msg = llm.invoke(
        f"""
        Student Name: {state['name']}
        Grade: {state['grade']}/20

        Give 3 practical study recommendations.
        """
    )

    return {"recommendation": msg.content}

def aggregator(state: State):
    """Generate a professional final report"""

    msg = llm.invoke(
        f"""
        Create a professional student report using:

        Evaluation:
        {state['evaluation']}

        Motivation:
        {state['motivation']}

        Recommendations:
        {state['recommendation']}

        Format it clearly with sections and a final summary.
        """
    )

    return {"report": msg.content}

workflow_builder = StateGraph(State)

workflow_builder.add_node("grade_evaluator",grade_evaluator)
workflow_builder.add_node("motivation_generator",motivation_generator)
workflow_builder.add_node("recommendation_generator",recommendation_generator)
workflow_builder.add_node("aggregator",aggregator)

workflow_builder.add_edge(START, "grade_evaluator")
workflow_builder.add_edge(START, "motivation_generator")
workflow_builder.add_edge(START, "recommendation_generator")

workflow_builder.add_edge("grade_evaluator", "aggregator")
workflow_builder.add_edge("motivation_generator", "aggregator")
workflow_builder.add_edge("recommendation_generator", "aggregator")
workflow_builder.add_edge("aggregator", END)

workflow = workflow_builder.compile()

display(Image(workflow.get_graph().draw_mermaid_png()))

input_data = {
    "name": "Ahmed",
    "grade": 14.5
}

state = workflow.invoke(input_data)

print(state['report'])