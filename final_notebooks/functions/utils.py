from langchain.tools import Tool, StructuredTool
from langchain_experimental.utilities import PythonREPL
import os
from . import hf_config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder


def get_llm():
    HF_TOKEN = hf_config.get_hf_token()
    model_name = os.getenv("LLM_MODEL", "openai/gpt-oss-120b:fireworks-ai")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    timeout_s = int(os.getenv("LLM_TIMEOUT_S", "60"))

    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
        model=model_name,
        temperature=temperature,
        timeout=timeout_s,
    )
    return llm

def get_prompt():
    return ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in ocean/climate sciences. You have some knowledge about available datasets which you can find using the advisor tool. "
        "Based off of the advisor tools recommendation on dataset, you can use the loading tool to download some data. The loading tool will return the path "
        "to where the data is saved. After that completes, you should do your analysis, preferring to use the recommended code from the advisor, but writing your "
        "own with the python repl tool if it doesn't give you enough information. Do not make up any data. If you don't have data available to satisfy the user's request, "
        "return early and say so. After you display your analysis, give a brief description. "
        "Always produce and consume STRICT JSON for tool inputs/outputs where requested. Do not use the MUR SST dataset.")
        ,
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
