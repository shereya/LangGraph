import langgraph.graph import StateGraph
from typing import TypedDict
class State(TypedDict):
    a:int
    b:int
    result:int
def addition(state:State):
    return {"result":state["a"]+state["b"]}
def multiply(state:State):
    return {"result":state["result"]*2}
graph = StateGraph(State)
graph.add_node("A",addition)
graph.add_node("B",multiply)
graph.set_entry_point("A")
graph.add_edge("A","B")
app = graph.compile()
c = app.invoke({"a":5,"b":6,"result":0})
print(c)
