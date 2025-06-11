from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict
import os


load_dotenv()
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
    openai_api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name="gpt-4.1",  
    temperature=0.2  
)

class State(TypedDict):
    messages: Annotated[list,add_messages]  
    # messages will be of type list and whenever we want to change message we can use add_messages


# we define a graph builder
graph_builder= StateGraph(State)

# nodes of the graph
def chatbot(state:State)->State:
    return {"messages":[llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot") # adding an edge from START to chatbot node
graph_builder.add_edge("chatbot", END) # adding an edge from chatbot node to END (basically making a flow from start-> chatbot -> end)

graph=graph_builder.compile()

user_input=input("Enter your message: ")
state=graph.invoke({"messages":[{"role": "user", "content": user_input}]})

print(state["messages"][-1].content)  # Print the response from the chatbot

# response=llm.invoke("What is the capital of France?")
# print(response.content)
