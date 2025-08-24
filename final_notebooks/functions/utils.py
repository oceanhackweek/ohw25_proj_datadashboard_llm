from langchain.tools import Tool, StructuredTool
from langchain_experimental.utilities import PythonREPL
import os
from . import hf_config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder


def get_llm(token: str):

    if not token:
        HF_TOKEN = hf_config.get_hf_token()
    else:
        HF_TOKEN = token
        
    
    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        model="openai/gpt-oss-120b:fireworks-ai" 
    )
    return llm

def get_prompt():
    return ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a methodical and precise climate data science orchestrator. Your job is to follow a strict three-step workflow to answer user requests. Do not deviate from this workflow.
            
            [YOUR WORKFLOW]
            
            **Step 1: Advise**
            - Use the `advisor_tool` to identify the best dataset and variable(s) for the user's request.
            - If the advisor finds no suitable data, you MUST stop and inform the user that their request cannot be fulfilled. DO NOT continue

            
            **Step 2: Analyze Data**
            - based on the `advisor_tool`'s suggestion, read the appropriate file.
            - Use the `python_repl` tool to write and execute code for the analysis.
            - After the code generates output (like a plot), provide a brief, clear description of the result.
            
            ---
        """
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
