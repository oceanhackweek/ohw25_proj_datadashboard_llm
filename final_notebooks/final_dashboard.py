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
    os.chdir(path='ohw25_proj_datadashboard_llm/final_notebooks')
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
    return hf_config, load_agent_executor, mo, ol


@app.cell
def _(load_agent_executor):
    token = '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'
    executor = load_agent_executor(token)
    return (token,)


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
    return


@app.cell
def _(hf_config, load_agent_executor, mo, token, widget):
    def my_model2(messages, widget):
        question = messages[-1].content   
        my_token = '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'
        hf_config.set_hf_token(my_token)
        executor = load_agent_executor(token)

        map_frame = widget.value["view_state"]["extent"]
        point_selected = widget.value.get("clicked", {}).get("coordinate", [0, 0])
        if point_selected != [0,0]:
            point_selected = point_selected + [point_selected[0] + 1, point_selected[1] + 1]

        # Create the executor
        remark = f'If the {map_frame} or {point_selected} is not [0,0], answer questions about the data from this selected area.'
        result = executor.invoke({"input": question + remark})
        return result

    mo.ui.chat(lambda messages: my_model2(messages, widget))
    return


if __name__ == "__main__":
    app.run()
