# %%
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import Annotated, TypedDict, Literal
from operator import add

# %%
API_KEY = "USE YOUR API KEY"

# %%
model = ChatGroq(model="openai/gpt-oss-120b",api_key=API_KEY)

# %%
class postState(TypedDict):
    post_text:str
    category: Annotated[list[Literal["Tutorial","News","Opinion"]],"Among the given topics, choose the category that the post is based on. It can be more than one category"]
    tut_sum : str
    news_sum : str
    op_sum : str
    final_sum :str
    score:Annotated[int, "Score out of 10"]
    iteration:int
    max_iteration:int    
    history : Annotated[list[str], add]
    
    

# %%
def find_categories(state:postState)->postState:
    prompt = f'''Find the categories that the text of the post belongs to. 
    I do not need extra explanation. Just the names of the categories would do. 
    The categories can be either "Tutorial", "News" and "Opinion". 
    The post can belong to more than one category. 
    This is the post - {state["post_text"]}'''
    model_response = model.invoke(prompt).content
    categories = [cat.strip() for cat in model_response.split(",")]
    return {"category": categories}

# %%
def tutorial_summary(state:postState)->postState:
    prompt = f'''You are an editor covering and editing tutorial related stories.
    You have to generate a concise summary for the post in 150 words.
    This is the text of the post - {state["post_text"]}'''
    model_response = model.invoke(prompt).content
    return {"tut_sum":model_response}

# %%
def news_summary(state:postState)->postState:
    prompt = f'''You are an editor covering and news tutorial related stories.
    You have to generate a concise summary for the post in 150 words.
    This is the text of the post - {state["post_text"]}'''
    model_response = model.invoke(prompt).content
    return {"news_sum":model_response}

# %%
def opinion_summary(state:postState)->postState:
    prompt = f'''You are an editor covering and editing tutorial related stories.
    You have to generate a concise summary for the post in 150 words.
    This is the text of the post - {state["post_text"]}'''
    model_response = model.invoke(prompt).content
    return {"op_sum":model_response}

# %%
def route_summary(state:postState)->Literal["tutorial_summary","opinion_summary","news_summary"]:
    routing = []
    for i in state["category"]:
        if i.strip().lower()=="tutorial":
            routing.append("tutorial_summary")
        if i.strip().lower()=="news":
            routing.append("news_summary")
        if i.strip().lower()=="opinion":
            routing.append("opinion_summary")
    return routing
        
    

# %%
def final_summary(state:postState)->postState:
    summary_text = state.get("tut_sum", "") + state.get("news_sum", "") + state.get("op_sum", "")
    prompt = f'''Make a TLDR summary from the summary. Keep it between 100-150 words
    {summary_text}'''
    model_response = model.invoke(prompt).content
    return {'final_sum':model_response}

# %%
def quality_score(state:postState)->postState:
    prompt = f'''As a strict evaluator, give the final summary generated a score out of '\10'.
    The final score cannot be 10. 
    Give me only the score and not any other extra explanation.
    {state["final_sum"]}'''
    model_response = model.invoke(prompt).content
    return {
        "score":int(model_response),
        "iteration":state["iteration"]+1,
        "history": [str(state["final_sum"])]
    }

# %%
def check_evaluation(state:postState)->str:
    if(state["iteration"]>state["max_iteration"] or state["score"]>7):
        return "Approved"
    else:
        return "Not Approved"

# %%
graph = StateGraph(postState)

graph.add_node("find_categories",find_categories)
graph.add_node("tutorial_summary",tutorial_summary)
graph.add_node("news_summary",news_summary)
graph.add_node("opinion_summary",opinion_summary)
graph.add_node("final_summary",final_summary)
graph.add_node("quality_score",quality_score)



graph.add_edge(START,"find_categories")
graph.add_conditional_edges("find_categories",route_summary)
graph.add_edge("tutorial_summary","final_summary")
graph.add_edge("news_summary","final_summary")
graph.add_edge("opinion_summary","final_summary")
graph.add_edge("final_summary","quality_score")
graph.add_conditional_edges("quality_score",check_evaluation, 
{
    "Approved":"final_summary",
    "Not Approved":END 
})

workflow = graph.compile()
workflow

# %%
workflow.invoke({"post_text":'''Breaking: Tech Giant Announces Major Layoffs

In a shocking announcement made earlier today, TechCorp revealed plans to lay off 15,000 employees globally, representing approximately 8% of its workforce. The decision comes amid declining revenue and increased pressure from shareholders.

The company's CEO stated in an internal memo that the restructuring is necessary to "streamline operations and focus on core profitable segments." Wall Street responded positively, with stock prices rising 3% following the news.

Industry analysts predict this could trigger a domino effect across the tech sector, as several other major companies are reportedly considering similar cost-cutting measures. The layoffs are expected to be completed by March 2025.

My Take: This is a short-sighted and cruel decision. While shareholders celebrate their quarterly gains, thousands of families face uncertainty and financial hardship during the holiday season. 

I believe this reflects a broader problem in corporate America—prioritizing profit margins over human welfare. These aren't just "headcount reductions" on a spreadsheet; they're real people with mortgages, children, and dreams.

The tech industry has long prided itself on innovation and disruption, but perhaps it's time we disrupted the outdated notion that mass layoffs are an acceptable solution to economic challenges. Companies with billion-dollar valuations can surely find more compassionate alternatives.

In my opinion, history will judge this era harshly for treating workers as disposable resources rather than valued contributors to success.
''', "iteration":0, "max_iteration":3})
