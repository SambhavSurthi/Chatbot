from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,BaseMessage,AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
from typing import TypedDict,Annotated,List
import sqlite3

load_dotenv()

# Persistence Creation
chat_history=sqlite3.connect('chat_history.db', check_same_thread=False)
checkpoint=SqliteSaver(conn=chat_history)


# State Creation
class ChatState(TypedDict):
    messages:Annotated[List[BaseMessage],add_messages]
    
    
# chat model initilization

chat_model=ChatGroq(model='llama-3.1-8b-instant')    

# state Functions
def chat(state:ChatState):
    message=state['messages']
    response=chat_model.invoke(input=message)
    return {'messages':response}


# graph Creation
graph=StateGraph(state_schema=ChatState)

graph.add_node('chat',chat)

graph.add_edge(START,'chat')
graph.add_edge('chat',END)

workflow=graph.compile(checkpointer=checkpoint)

# while True:
#     config={'configurable': {'thread_id': '1'}}
#     user_input=input("User: ")
#     if user_input.lower() in ['exit','quit']:
#         break
#     response=workflow.invoke(config=config, input={'messages': [HumanMessage(content=user_input)]})
#     print(f"AI: {response['messages'][-1].content}")
    
# print(workflow.get_state(config=config))
