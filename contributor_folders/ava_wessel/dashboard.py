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
    from ipywidgets import Output
    from IPython.display import display
    from shapely.geometry import box
    from ipyleaflet import Map, DrawControl

    from langchain_community.vectorstores import Chroma
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    return ChatOpenAI, ChatPromptTemplate, Draw, StrOutputParser, folium, mo


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
def _():
    import json
    from shapely.geometry import shape, Point

    # Load the saved GeoJSON
    with open('~/Downloads/map_selection.geojson') as f:
        geojson_data = json.load(f)

    # Assuming only one feature (your rectangle)
    polygon_geom = shape(geojson_data['features'][0]['geometry'])

    return


@app.cell
def _(ChatOpenAI, ChatPromptTemplate, StrOutputParser, mo):
    def my_model(messages, config):
        HF_TOKEN = "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"

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

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({"question": "{question}"})



        return response

    mo.ui.chat(my_model)
    return


if __name__ == "__main__":
    app.run()
