# %%
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal, List
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq



# %%
#API_KEY = USE YOUR API KEY

# %%
#model = ChatOllama(model="llama3.2")
model = ChatGroq(model="openai/gpt-oss-120b", api_key=API_KEY)

# %%
eval_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=API_KEY)

# %%
class ticketState(TypedDict):
    ticket_text : str
    category : Annotated[Literal["Billing", "Technical", "General"], "Choose the ticket category."]
    final_solution:str
    quality_score: Annotated[int, "Give the resolution a score out of 10"]
    response:str
    iteration: int 
    max_iteration:int
    
    

# %%
def classify_ticket(state:ticketState) -> ticketState:
    prompt = f"""
Classify the ticket into ONLY ONE of the following categories:
Billing, Technical, General.

Return ONLY the category name.
Do NOT explain. Do NOT add extra text.
Ticket:
{state["ticket_text"]}
"""
    model_response = model.invoke(prompt).content
    return {"category":model_response}

# %%
def technical_solution(state:ticketState) -> ticketState:
    prompt = f"""
    As someone who provides solutions for technical issues exclusively, 
    give a solution for the given ticket, {state["ticket_text"]}"""
    model_response = model.invoke(prompt).content 
    return {'final_solution':model_response}


# %%
def billing_solution(state:ticketState) -> ticketState:
    prompt = f"""
    As someone who provides solutions for billing issues exclusively, 
    give a solution for the given ticket, {state["ticket_text"]}"""
    model_response = model.invoke(prompt).content 
    return {'final_solution':model_response}

# %%
def general_solution(state:ticketState) -> ticketState:
    prompt = f"""
    As someone who provides solutions for general issues exclusively, 
    give a solution for the given ticket, {state["ticket_text"]}"""
    model_response = model.invoke(prompt).content 
    return {'final_solution':model_response}

# %%
def route_ticket(state:ticketState)->Literal["technical_solution","general_solution","billing_solution"]:
    if(state["category"].lower().strip()=="technical"):
        return "technical_solution"
    elif(state["category"].lower().strip()=="general"):
        return "general_solution"
    else:
        return "billing_solution"

# %%
def generate_response(state:ticketState)->ticketState:
    prompt = f'''Now taking the solution generated, 
    draft a message being apologetic wherever needed and 
    develop a user-friendly response to the person who raised the ticket
    with clear and easy steps for them to follow to resolve the issue. 
    This is the solution generated. Just the email content will do {state["final_solution"]}'''
    model_response = model.invoke(prompt).content
    return {"response":model_response}

# %%
def quality_check(state:ticketState)->ticketState:
    prompt = f"""
Evaluate the response using the criteria below.

Scoring rules:
- Depth of understanding: max 3
- Politeness: max 2
- Effectiveness: max 5

IMPORTANT:
Return ONLY ONE NUMBER between 0 and 10.
Do NOT include '\10'.
Do NOT explain.
Do NOT add text.

Response:
{state['response']}
"""

    score = eval_model.invoke(prompt).content
    return{
        'quality_score': int(score),
         "iteration": state["iteration"]+1
    }


# %%
def check_evaluation_score(state: ticketState) -> str:
    if state["quality_score"] < 9 and state["iteration"] < state["max_iteration"]:
        return "generate_response"   
    else:
        return "__end__"    


# %%
'''def quality_check(state:ticketState)->ticketState:
    prompt = f"""
Evaluate the response using the criteria below.

Scoring rules:
- Depth of understanding: max 3
- Politeness: max 2
- Effectiveness: max 5

IMPORTANT:
Return ONLY ONE NUMBER between 0 and 10.
Do NOT include '\10'.
Do NOT explain.
Do NOT add text.

Response:
{state['response']}
"""

    score = eval_model.invoke(prompt).content
    print(f"RAW SCORE FROM MODEL: '{score}'")  # See exact model output
    print(f"CURRENT ITERATION: {state['iteration']}")
    
    return {
        'quality_score': int(score),
        'iteration': state['iteration'] + 1 
    }

def check_evaluation_score(state: ticketState) -> str:
    print(f"CHECKING - Score: {state['quality_score']}, Iteration: {state['iteration']}, Max: {state['max_iteration']}")
    print(f"Condition 1 (score < 9): {state['quality_score'] < 9}")
    print(f"Condition 2 (iter < max): {state['iteration'] < state['max_iteration']}")
    
    if state["quality_score"] < 9 and state["iteration"] < state["max_iteration"]:
        print("RETURNING: generate_response")
        return "generate_response"   
    else:
        print("RETURNING: __end__")
        return "__end__"'''

# %%
graph = StateGraph(ticketState)

graph.add_node('classify_ticket',classify_ticket)
graph.add_node('technical_solution',technical_solution)
graph.add_node('billing_solution',billing_solution)
graph.add_node('general_solution',general_solution)
graph.add_node('generate_response',generate_response)
graph.add_node('quality_check',quality_check)

graph.add_edge(START,'classify_ticket')
graph.add_conditional_edges('classify_ticket',route_ticket)
graph.add_edge('technical_solution','generate_response')
graph.add_edge('billing_solution','generate_response')
graph.add_edge('general_solution','generate_response')
graph.add_edge('generate_response','quality_check')
graph.add_conditional_edges(
    'quality_check',
    check_evaluation_score,
    {"generate_response": "generate_response", "__end__": END}
)
workflow = graph.compile()
workflow

# %%
workflow.invoke({
    "ticket_text": "My refund hasn’t been credited yet",
    "iteration": 0,
    "max_iteration": 3
})


# %% 
result

# %%
result["response"]
#

