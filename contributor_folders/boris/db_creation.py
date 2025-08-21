import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import hf_config


def read_examples(filename: str):
    """
    Reads a JSON file containing code description examples
    and returns them as a list of dictionaries.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

examples = read_examples("code_descriptions.json")
doc_embedder = HuggingFaceEndpointEmbeddings(
                                        model="Qwen/Qwen3-Embedding-8B",
                                        task="feature-extraction",
                                        model_kwargs={"normalize": True},
                                        hhuggingfacehub_api_token=hf_config.get_hf_token()
                                    )
def create_db_examples():

    docs = []
    for doc in examples:
        # use the definition as the content, and keep the term (and letter) as metadata
        docs.append(
            Document(
                page_content=doc['page_content'],
                metadata=doc['metadata']  
            )
        )
    vector_store_hf = Chroma.from_documents(
                                            docs,
                                            doc_embedder,
                                            persist_directory="./chroma_db_examples",
                                        )
    vector_store_hf.persist()
    return vector_store_hf