# %%
from langgraph.graph import StateGraph, START, END
from typing import Annotated, TypedDict
from operator import add
from langchain_groq import ChatGroq


# %%
API_KEY = "YOUR API KEY"

# %%
model = ChatGroq(model="openai/gpt-oss-120b", api_key=API_KEY)

# %%
class reflexState(TypedDict):
    topic : str
    history : Annotated[list[str],add]
    thought : Annotated[list[str],add]
    iteration : int
    max_iteration : int
    score : int

# %%
def generate_content(state:reflexState)->reflexState:
    if(len(state["thought"])==0):
        prompt = f'''Generate a 150-200 word essay on the given topic - {state["topic"]}
        Give me only the essay. Just the essay. Nothing else. Give a bad quality response'''
        model_response = model.invoke(prompt).content 
        return {'history':[str(model_response)]}
    else:
        prompt = f'''Use the feedback recieved to rewrite the essay on the topic - {state["topic"]}. 
        Use 150-200 words. Give me only the essay. Just the essay. Nothing else. 
        The feedback for the previously generated essay is as follows - {state["thought"][-1]}'''
        model_response = model.invoke(prompt).content 
        return {'history':[str(model_response)]}

# %%
def quality_check(state:reflexState)->reflexState:
    prompt_1 = f'''I will give you an essay to evaluate, give me your honest feedback. 
    The parameters for evaluation are 
    1. Relevance
    2. Fluency
    3. Grammar
    Just the feedback will do. 
    {state["history"][-1]}'''
    response_1 = model.invoke(prompt_1).content
    prompt_2 = f'''Based on the feedback recieved, give the new content a score out of "\10". The score cannot be 10.
    Give me only the score nothing else. Feedback: {response_1}'''
    response_2 = model.invoke(prompt_2).content
    return {"thought":[response_1],"score":int(response_2),"iteration": state["iteration"] + 1}


# %%
def evaluate_essay(state:reflexState):
    if(state["score"]>8 or state["iteration"]>state["max_iteration"]):
        return "Approved"
    else:
        return "Not Approved"

# %%
graph = StateGraph(reflexState)
graph.add_node("generate_content",generate_content)
graph.add_node("quality_check",quality_check)
graph.add_edge(START, "generate_content")
graph.add_edge("generate_content","quality_check")
graph.add_conditional_edges("quality_check",evaluate_essay,
    {"Not Approved":"generate_content", 
     "Approved":END})
workflow = graph.compile()
workflow

# %%
result = workflow.invoke({"topic":"India - The Global Concert Hub", "iteration":0, "max_iteration":3})

# %%
result

# %%
result["history"]

# %%
result["thought"]


