import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    import folium
    from folium.plugins import Draw
    import geopandas as gpd
    from shapely.geometry import box
    from IPython.display import display
    import json

    from langchain_community.vectorstores import Chroma
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    return (
        ChatOpenAI,
        ChatPromptTemplate,
        Draw,
        StrOutputParser,
        folium,
        json,
        mo,
    )


@app.cell
def _(mo):
    mo.md("<h1>Data Dashboard with Chatbot!").center()
    return


@app.cell
def _(mo):
    mo.md("<h0> OceanHackWeek 2025").center()
    return


@app.cell
def _(Draw, folium):
    m = folium.Map(location=[0, 0], zoom_start=2)
    draw = Draw(export= True,  filename='map_selection.geojson',
        draw_options={
            'polyline': False,
            'polygon': False,
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'rectangle': True
        },
        edit_options={'edit': False}
    )
    draw.add_to(m)

    m
    return


@app.cell
def _(json):
    def read_json_file(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    return


@app.cell
def _(
    ChatOpenAI,
    ChatPromptTemplate,
    StrOutputParser,
    geojson_file,
    mo,
    summarize_geojson,
):
    def my_model(messages, config):
        HF_TOKEN = "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"

        polygon_summary = ""
        if geojson_file:
            polygon_summary = summarize_geojson(geojson_file)

        llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
            model="openai/gpt-oss-20b:fireworks-ai"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant. "
                       "You have access to a map selection polygon "
                       "that the user can draw."),
            ("human", "{question}")
        ])
        allow_attachments=['map_selection.geojson']

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({"question": "{question}"})



        return response

    mo.ui.chat(my_model, allow_attachments=['map_selection.geojson'])
    return


if __name__ == "__main__":
    app.run()
