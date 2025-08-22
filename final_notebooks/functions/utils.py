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
            - If the advisor finds no suitable data, you MUST stop and inform the user that their request cannot be fulfilled.
            
            **Step 2: Load Data**
            - Use the `loader_tool` with the exact dataset and variable names from Step 1.
            - This tool will return a local `file_path` (e.g., "temp/data.nc"). This path is critical for the next step.
            
            **Step 3: Analyze Data**
            - Use the `python_repl` tool to write and execute code for the analysis.
            - Follow the critical rule below.
            - After the code generates output (like a plot), provide a brief, clear description of the result.
            
            ---
            [CRITICAL RULE FOR STEP 3: ANALYSIS]
            
            **When you use the `python_repl` tool, your code MUST use the `file_path` provided by the `loader_tool` from Step 2.**
            
            - Your Python code should always begin by opening this specific path (e.g., `ds = xarray.open_dataset("temp/data.nc")`).
            - **DO NOT** attempt to re-download, re-load, or access data from any other source or path inside the `python_repl` tool. The data is already prepared for you.
        """
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
