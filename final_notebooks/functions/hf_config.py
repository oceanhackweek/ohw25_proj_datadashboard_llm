# config.py
HF_TOKEN = None  # Default is None, set it from Jupyter or script

def set_hf_token(token: str):
    """
    Set HuggingFace token globally.
    """
    global HF_TOKEN
    HF_TOKEN = token

def get_hf_token() -> str:
    """
    Get HuggingFace token (raises error if not set).
    """
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN is not set. Please run config.set_hf_token('<your_token>').")
    return HF_TOKEN