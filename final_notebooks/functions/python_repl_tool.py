from langchain_experimental.utilities import PythonREPL
from langchain.tools import Tool

def create_python_repl():
    python_repl = PythonREPL()
    
    
    python_repl_tool = Tool(
        name="python_repl",
        func=python_repl.run,
        description="""
        You are a Python REPL specialized for scientific data analysis.
    
        You receive:
        - A file path to the downloaded dataset. This path comes from the load_climate_data tool. Do not use any other path.
        - Optionally, an example analysis function from advisor Tool that may be relevant to the query.
        
        NOTE: do NOT download any new data from anywhere. You should directly access the data at the path that is given to you.
        
        Your job:
        1. If an example function is provided and fits the user’s query, use it directly with the dataset. Make sure you load the data!
        2. If no suitable example function is provided, write your own analysis code using standard scientific Python packages such as:
           - xarray (for handling datasets)
           - matplotlib (for plotting, with clean labels and colorbars)
           - numpy (for computations)
           - cartopy or geopandas (if maps are needed)
           - cmocean or matplotlib colormaps (for nice scientific colormaps)
        3. Keep your code clean, minimal, and runnable in a single Python cell.
        4. If user’s request cannot be completed, return a helpful error message explaining why.
        5. After you are done showing your result, please give an explanation of what the user is seeing.
        
        Guidelines:
        - Always open the dataset using xarray from the provided file path.
        - Assume the dataset may be large: use efficient operations (`.sel`, `.isel`, `.mean`, `.plot`, etc.).
        - Include titles, axis labels, and legends where appropriate.
        - Prefer clarity and readability of code over cleverness.
        - Do not invent dataset fields—only use those provided in the dataset info.
        - If the user asks for a time range that you might think is too large, warn them and suggest re-execution.
    
        """,
    )
    return python_repl_tool