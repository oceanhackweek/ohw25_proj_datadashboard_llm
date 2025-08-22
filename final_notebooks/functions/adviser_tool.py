import os, json
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError
from . import hf_config
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import Chroma

SYSTEM_PROMPT = """
Pick the best single dataset and variable(s) using only the task description.

[TASK DESCRIPTION]
{safe_desc}

[OUTPUT SCHEMA — return STRICT JSON matching the schema below]
{
  "dataset": "<dataset name or none>",
  "variables": ["<variable names>"],
  "lat_lon_bounds": {
    "lat": [<lat_min>, <lat_max>] | "global",
    "lon": [<lon_min>, <lon_max>] | "global"
  },
  "time_range": "<YYYY-MM-DD to YYYY-MM-DD>" | "full available",
  "suggestions": "<from description only>"
}

[DECISION RULES]
- Choose the most specific dataset & variables explicitly supported by the description.
- If region/time are missing, use "global" and "full available".
- If no suitable match exists, set dataset to "none" and variables to an empty list.
- Suggestions must reflect ONLY what the description states (no external inference).
- For now, do NOT use the mur sst dataset.
- Respond with ONLY JSON. No extra text.
"""

class AdviserResult(BaseModel):
    dataset: str = Field(..., description="Dataset name or 'none'")
    variables: list[str] = Field(default_factory=list, description="List of variable names")
    lat_lon_bounds: dict = Field(..., description="{""lat"": [min,max] or 'global', ""lon"": [min,max] or 'global'}")
    time_range: str = Field(..., description="'<start> to <end>' or 'full available'")
    suggestions: str = Field(..., description="Notes from description only")
    example_paths: list[str] = Field(default_factory=list, description="Top-k example file paths relevant to the query")

def load_safe_desc(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return text

def get_example_of_visualizations(query: str, k: int = 3) -> list[str]:
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
    results = vector_store_hf.similarity_search_with_score(query, k=k)
    paths: list[str] = []
    for doc, _ in results:
        file_name = doc.metadata.get('source', '').lstrip('./')
        if file_name:
            full_path = os.path.join('./', file_name)
            paths.append(full_path)
    return paths

def description_reader(query: str):
    safe_desc = load_safe_desc("functions/datasets.json")
    llm = ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_config.get_hf_token(),
        model="openai/gpt-oss-20b:fireworks-ai"
    )

    system_message = SYSTEM_PROMPT.replace("{safe_desc}", safe_desc)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": query})

    # Try to parse and validate; if it fails, fall back to a minimal object
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "dataset": "none",
            "variables": [],
            "lat_lon_bounds": {"lat": "global", "lon": "global"},
            "time_range": "full available",
            "suggestions": "none"
        }

    # Inject example paths
    example_paths = get_example_of_visualizations(query)
    data["example_paths"] = example_paths

    try:
        validated = AdviserResult(**data)
        return json.dumps(validated.model_dump())
    except ValidationError:
        # Fallback to safe minimal response with examples
        fallback = AdviserResult(
            dataset="none",
            variables=[],
            lat_lon_bounds={"lat": "global", "lon": "global"},
            time_range="full available",
            suggestions="none",
            example_paths=example_paths,
        )
        return json.dumps(fallback.model_dump())

class AdviserParams(BaseModel):
    query: str = Field(..., description="User query")

def create_adviser_tool():

    adviser_tool = StructuredTool.from_function(
        description_reader,
        name="adviser_tool",
        description="Use this tool to find a suitable dataset and code example. Returns a JSON string matching AdviserResult.",
        args_schema=AdviserParams,
    )
    return adviser_tool