import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    import openlayers as ol
    from shapely.geometry import box
    import os

    os.chdir(path='/home/jovyan/ohw25_proj_datadashboard_llm/final_notebooks')
    from langchain_community.vectorstores import Chroma
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    from langchain_openai import ChatOpenAI
    from langchain.prompts import MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    import json
    import hf_config
    from executor import load_agent_executor


    current_directory = os.getcwd()
    print(current_directory)
    from dotenv import load_dotenv
    return hf_config, load_agent_executor, mo, ol, os


@app.cell
def _(mo):
    mo.md(
        """
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; background: #f0f9ff; padding: 5px; border-radius: 10px;">
        <img src="https://oceanhackweek.org/_static/logo.png" 
             alt="Logo" width="60" height="60">
        <span style="font-family: monospace; 
                     font-size: 48px; 
                     font-weight: 700; 
                     color: black; 
                     line-height: 0.5;">
            SplashBot
        </span>
        1.0
        <img src="https://cdn-icons-png.flaticon.com/512/6501/6501379.png" 
             alt="Logo" width="100" height="90">

    </div>
    """
    )
    return


@app.cell
def _(ol):
    m = ol.MapWidget()
    m.add_click_interaction()
    return (m,)


@app.cell
def _(m, mo):
    widget = mo.ui.anywidget(m)
    widget
    return (widget,)


@app.cell
def _(mo):
    mo.md(
        r"""
    <div style="gap: 30px;">
      <h1 style="font-family: Arial, sans-serif; font-size: 15px; color: lightseagreen; margin: 0; 
                 background: #f0f9ff; padding: 8px; font-weight: bold; border-radius: 10px;
                 display: flex; align-items: center; gap: 10px;">
        <img src="https://cdn-icons-png.flaticon.com/512/2976/2976128.png" 
             alt="Logo" width="25" height="0">
        Explore the map by dragging and zooming, or click any location to analyze its data. Use the chatbot for instant insights and comparisons.
        </h1>
      </h1>
    </div>
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    <div style="gap: 12px; background: #f0f9ff; padding: 8px; border-radius: 10px;">
        <h1 style="font-family: Arial, sans-serif; font-size: 17px; color: black; margin: 0; font-weight: bolder">
            Your Selected Data Boundaries:
        </h1>
    </div>
    """
    )
    return


@app.cell
def _(widget):
    x , y = 0, 0
    point_selected2 = widget.value.get("clicked", {}).get("coordinate", [0, 0])
    if point_selected2:
        x, y = widget.value.get("clicked", {}).get("coordinate", [0, 0])
    return x, y


@app.cell
def _(mo, widget, x, y):
    map_data = widget.value["view_state"]
    map_frame = map_data['extent']
    mo.md(
        f"<span style='font-size:16px'><b>Longitude min:</b> {round(map_frame[0], 2)}, "
        f"<b>Longitude max:</b> {round(map_frame[2], 2)}</span><br>"
        f"<span style='font-size:16px'><b>Latitude min:</b> {round(map_frame[1], 2)}, "
        f"<b>Latitude max:</b> {round(map_frame[3], 2)}</span><br>"
        f"<span style='font-size:18px ; color:darkblue'><b>Point Selected:</b> {round(x, 2)}, {round(y, 2)}</span>"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    <div style="gap: 12px;">
      <h1 style="font-family: Arial, sans-serif; font-size: 15px; color: lightseagreen; margin: 0; 
                 background: #f0f9ff; padding: 8px; font-weight: bold; border-radius: 10px;
                 display: flex; align-items: center; gap: 10px;">
        <img src="https://cdn-icons-png.flaticon.com/512/2976/2976128.png" 
             alt="Logo" width="25" height="40">
        Please provide HF token to run the Chatbot.
      </h1>
    </div>
    """
    )
    return


@app.cell
def _(mo):
    text_area = mo.ui.text_area(placeholder="Enter HF Token ...", )
    text_area
    return (text_area,)


@app.cell
def _(text_area):
    user_key = text_area.value.strip()
    return (user_key,)


@app.cell
def _(hf_config, load_agent_executor, mo, os, user_key, widget):

    import re
    from PIL import Image
    import glob

    def clear_figures_folder():
        """Delete all PNG files from the figures_temp directory"""
        if os.path.exists('figures_temp'):
            png_files = glob.glob('figures_temp/*.png')
            for file_path in png_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            print(f"Cleared {len(png_files)} files from figures_temp")

    def my_model2(messages, widget):    
        question = messages[-1].content    

        # Clear all existing figures at the start of each new question
        clear_figures_folder()

        my_token = user_key
        hf_config.set_hf_token(my_token)    
        executor = load_agent_executor(my_token)    
        map_frame = widget.value["view_state"]["extent"]    
        point_selected = widget.value.get("clicked", {}).get("coordinate", [0, 0])    

        if point_selected != [0,0]:        
            point_selected = point_selected + [point_selected[0] + 1, point_selected[1] + 1]    

        # Create the executor    
        remark = f'If the {map_frame} or {point_selected} is not [0,0], answer questions about the data from this selected area.'    
        result = executor.invoke({"input": question + remark})

        # Extract figure paths from this execution
        figure_paths = extract_figure_paths(result)

        # Create response with figures
        response_content = []

        # Add the text response
        response_content.append(mo.md(str(result)))

        # Show only the last (most recent) figure generated in this execution
        if figure_paths:
            last_figure_path = figure_paths[-1]  # Get the most recent figure
            response_content.append(mo.md("### Generated Figure:"))
            if os.path.exists(last_figure_path):
                try:
                    # Display only the last image
                    img = Image.open(last_figure_path)
                    response_content.append(mo.image(src=last_figure_path, alt=f"Generated plot: {os.path.basename(last_figure_path)}"))
                except Exception as e:
                    response_content.append(mo.md(f"Error loading image {last_figure_path}: {str(e)}"))
            else:
                response_content.append(mo.md(f"Figure not found: {last_figure_path}"))

        # Return combined content
        return mo.vstack(response_content) if response_content else result

    def extract_figure_paths(result_text):
        """Extract figure paths from the executor result and scan directory for new files"""
        # Convert result to string if it's not already
        result_str = str(result_text)

        # Multiple patterns to catch different ways figures might be mentioned
        patterns = [
            r'figures_temp/plot_\d+_\d+_\d+_fig\d+\.png',  # Original pattern
            r'figures_temp/[^/\s]+\.png',  # Any PNG in figures_temp
            r'Figure saved to:\s*([^\n\r]+\.png)',  # Saved to pattern
            r'([^\s]+\.png)',  # Any PNG file mentioned
            r'saved to:\s*([^\n\r]+)',  # General saved to pattern
        ]

        found_paths = []
        for pattern in patterns:
            matches = re.findall(pattern, result_str)
            found_paths.extend(matches)

        # Also scan the figures_temp directory for any new PNG files
        if os.path.exists('figures_temp'):
            all_pngs = glob.glob('figures_temp/*.png')
            found_paths.extend(all_pngs)

        # Clean up paths and filter for existing files with reasonable content
        valid_paths = []
        for path in found_paths:
            clean_path = path.strip()

            # Ensure it's a proper path
            if not clean_path.endswith('.png'):
                continue

            if os.path.exists(clean_path):
                # Check if the file has reasonable size (not empty plot)
                file_size = os.path.getsize(clean_path)
                if file_size > 1000:  # Lowered threshold - even small plots should be > 1KB
                    if clean_path not in valid_paths:
                        valid_paths.append(clean_path)

        # Sort by modification time to get chronological order
        if valid_paths:
            valid_paths.sort(key=lambda x: os.path.getmtime(x))

        return valid_paths

    # Create the chat interface
    mo.ui.chat(lambda messages: my_model2(messages, widget))
    return


if __name__ == "__main__":
    app.run()
