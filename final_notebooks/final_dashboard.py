import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    import geopandas as gpd
    import openlayers as ol
    from shapely.geometry import box
    import os
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


    current_directory = os.getcwd()
    print(current_directory)
    from dotenv import load_dotenv
    return (
        AgentExecutor,
        ChatOpenAI,
        ChatPromptTemplate,
        MessagesPlaceholder,
        create_tool_calling_agent,
        hf_config,
        mo,
        ol,
    )


@app.cell
def _(hf_config):
    # Set token once in your notebook
    my_token = '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'
    hf_config.set_hf_token(my_token)
    LANGSMITH_TRACING="true"
    LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
    LANGSMITH_API_KEY=''
    LANGSMITH_PROJECT="ohw_llm"
    return


@app.cell
def _():
    from db_creation import create_db_examples
    vector_store_hf = create_db_examples()
    return


@app.cell
def _(mo):
    mo.md(
        """
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px; background: #f0f9ff; padding: 5px; border-radius: 10px;">
        <img src="https://oceanhackweek.org/_static/logo.png" 
             alt="Logo" width="60" height="60">
        <span style="font-family: Arial, Helvetica, sans-serif; 
                     font-size: 48px; 
                     font-weight: 700; 
                     color: #002147; 
                     line-height: 1;">
            Data Dashboard with Chatbot
        </span>
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
    <div style="gap: 12px; background: #f0f9ff; padding: 8px; border-radius: 10px;">
        <h1 style="font-family: Arial, sans-serif; font-size: 12px; color: green; margin: 0; font-weight: bold">
            Explore the map by dragging and zooming, or click any location to analyze its data. Use the chatbot for instant insights and comparisons.
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
    <div style="gap: 12px; ">
        <h1 style="font-family: Arial, sans-serif; font-size: 12px; color: green; margin: 0; background:#f0f9ff; padding: 8px;font-weight: bold; border-radius: 10px; ">
            Please provide HF token to run the Chatbot
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
def _():
    return


@app.cell
def _(
    AgentExecutor,
    ChatOpenAI,
    ChatPromptTemplate,
    MessagesPlaceholder,
    create_tool_calling_agent,
    hf_config,
    mo,
    widget,
):
    def my_model2(messages, widget):
        question = messages[-1].content   
        my_token = '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'
        hf_config.set_hf_token(my_token)

        map_frame = widget.value["view_state"]["extent"]
        point_selected = widget.value.get("clicked", {}).get("coordinate", [0, 0])

        from adviser_tool import create_adviser_tool
        adviser_tool_llm = create_adviser_tool()
        tools = [adviser_tool_llm]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"You are an expert in climate data analysis, you have adviser tool, which can help you to asnwer user's questions about variables/datasets. If the question about data, use only information from adviser_tool. If the {map_frame} or {point_selected} is not [0,0], answer questions about the data from this selected area. "),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_config.get_hf_token(),
            model="openai/gpt-oss-120b:fireworks-ai"  
        )

        # Define the agent
        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
        )

        # Create the executor
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        result = agent_executor.invoke({"input": question})
        return result

    mo.ui.chat(lambda messages: my_model2(messages, widget))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
