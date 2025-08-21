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
    from langchain_core.output_parsers import StrOutputParser

    current_directory = os.getcwd()
    print(current_directory)
    return ChatOpenAI, ChatPromptTemplate, StrOutputParser, mo, ol


@app.cell
def _():
    from dotenv import load_dotenv
    load_dotenv()
    return


@app.cell
def _(mo):
    mo.md(
        """
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px;">
        <img src="https://oceanhackweek.org/_static/logo.png" 
             alt="Logo" width="50" height="50" style="border-radius: 0%;">
        <h1 style="font-family: Arial, sans-serif; font-size: 36px; color: black; margin: 0; font-weight: bold">
            Data Dashboard with Chatbot!
        </h1>
    </div>
    """
    )
    return


@app.cell
def _(mo):
 
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
    mo.md("""<h0>Your selected Data Boundaries:""")
    return


@app.cell
def _(mo, widget):
    map_data = widget.value["view_state"]
    map_frame = map_data['extent']
    mo.md(
        f"Longitude min: {round(map_frame[0], 2)}, Longitude max: {round(map_frame[2], 2)}  \n"
        f"Latitude min: {round(map_frame[1], 2)}, Latitude max: {round(map_frame[3], 2)}"
    )
    return


@app.cell
def _(mo):
    text_area = mo.ui.text_area(placeholder="Enter HF Token ...")
    text_area
    return (text_area,)


@app.cell
def _(text_area):
    user_key = text_area.value.strip()
    return (user_key,)


@app.cell
def _(ChatOpenAI, ChatPromptTemplate, StrOutputParser, mo, user_key, widget):
    def my_model(messages, widget):
        HF_TOKEN = user_key

        map_frame = widget.value["view_state"]["extent"]
        min_lon, min_lat, max_lon, max_lon = map_frame

        llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
            model="openai/gpt-oss-20b:fireworks-ai"
        )
        question = messages[-1].content

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant that will answer questions about the area selected in the map."
                       f"The current map boundaries are: {map_frame} that the user can draw."
            f"users can ask what current map boundaries are provided by {min_lon, min_lat, max_lon, max_lon}"),
            ("human", "{question}")
        ])

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({"question": "{question}"})

        return response

    mo.ui.chat(lambda messages: my_model(messages, widget))
    return


if __name__ == "__main__":
    app.run()
