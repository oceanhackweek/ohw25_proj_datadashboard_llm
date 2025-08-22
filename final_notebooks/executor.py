import functions.hf_config as hf_config
from functions.adviser_tool import create_adviser_tool
from functions.loader import create_loader_tool
from functions.python_repl_tool import create_python_repl
from functions.utils import get_llm, get_prompt
from langchain.agents import AgentExecutor, create_tool_calling_agent
from functions.db_creation import create_db_examples

def load_agent_executor(token: str):

    hf_config.set_hf_token(token)
    chroma = create_db_examples(token)
    advisor_tool = create_adviser_tool()
    loader_tool = create_loader_tool()
    repl_tool = create_python_repl()

    tools = [
        advisor_tool,
        loader_tool,
        repl_tool
    ]

    llm = get_llm(token)

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=get_prompt(),
    )
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)