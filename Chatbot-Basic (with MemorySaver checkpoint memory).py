# %%
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

# %%
API_KEY = "YOUR API KEY"

# %%
model = ChatGroq(model="openai/gpt-oss-120b", api_key=API_KEY)

# %%
class chatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

# %%
def chat_node(state:chatState)->chatState:
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages":[response]}

# %%
graph = StateGraph(chatState)
checkpointer = MemorySaver()
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)
workflow = graph.compile(checkpointer=checkpointer)
workflow

# %%
thread_id = '1'
while True:
    user_input = input("Enter text")
    if(user_input.strip().lower() in ["exit","quit","bye"]):
        break
    print('User Message:',user_input)
    print('AI Message:')
    config = {"configurable":{"thread_id":thread_id}}
    chat = workflow.invoke({"messages":[HumanMessage(user_input)]},config=config)
    print(chat["messages"][-1].content)


