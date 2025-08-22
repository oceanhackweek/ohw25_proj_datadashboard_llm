import os, json
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import hf_config
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

SYSTEM_PROMPT = """
 Pick the best single dataset and variable (might be several) using only the task description.
[TASK DESCRIPTION]
{safe_desc}

[OUTPUT SCHEMA — return ONLY these fields in this order]
dataset: <dataset name or "none">
variable: <variable name or "none">
lat,lon boundaries : <[lat_min, lat_max], [lon_min, lon_max] or "global">
time range: <YYYY-MM-DD to YYYY-MM-DD or "full available">
suggestions (from description only): <where this variable is available (region/coverage) if stated; else "none">

[DECISION RULES]
- Choose the most specific dataset & variables (can be more than one) explicitly supported by the description.
- If region/time are missing, use "global" and "full available".
- If no suitable match exists, set dataset and variable to "none".
- Suggestions must reflect ONLY what the description states (no external inference).
"""

def load_safe_desc(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return text.replace("{", "{{").replace("}", "}}")

def get_example_of_visualizations(query: str) -> str:
    doc_embedder = HuggingFaceEndpointEmbeddings(
                                        model="Qwen/Qwen3-Embedding-8B",
                                        task="feature-extraction",
                                        model_kwargs={"normalize": True},
                                        huggingfacehub_api_token=hf_config.get_hf_token()
                                    )
    vector_store_hf = Chroma(
        persist_directory="./chroma_db_examples",
        embedding_function=doc_embedder
    )
    results = vector_store_hf.similarity_search_with_score(query, k=1)
    doc, score = results[0]
    file_name = doc.metadata['source'].lstrip('./')
    full_path = os.path.join('./', file_name)

    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception:
        return ""

def description_reader(query: str):
    safe_desc = load_safe_desc("datasets.json")
    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_config.get_hf_token(),
        model="openai/gpt-oss-20b:fireworks-ai"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(safe_desc=safe_desc)),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query})
    example = get_example_of_visualizations(query)
    return response + '\n\nYou can use this code to analyse the data:\n\n' + example

class AdviserParams(BaseModel):
    query: str = Field(..., description="User query")

def create_adviser_tool():

    adviser_tool = StructuredTool.from_function(
        description_reader,
        name="adviser_tool",
        description="Use this tool to find a suitable dataset and code example",
        args_schema=AdviserParams,
    )
    return adviser_tool