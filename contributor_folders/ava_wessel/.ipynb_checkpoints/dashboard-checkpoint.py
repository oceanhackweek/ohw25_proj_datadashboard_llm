import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    import folium
    import leafmap.maplibregl as leafmap
    import geopandas as gpd
    from maplibre.plugins import MapboxDrawControls, MapboxDrawOptions
    from langchain_community.vectorstores import Chroma
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    return (
        ChatOpenAI,
        ChatPromptTemplate,
        MapboxDrawControls,
        MapboxDrawOptions,
        StrOutputParser,
        leafmap,
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
def _(leafmap):
    m = leafmap.Map(center=[37.77, -122.42], zoom=12, style="positron")
    draw_opts = MapboxDrawOptions(display_controls_default=True)
    draw_ctrl = MapboxDrawControls(draw_opts)
    m.add_control(draw_ctrl)
    m
    return m, draw_ctrl


@app.cell(hide_code=True)
def _(test):

    #tools for chatbot 
    map_data = test.draw_features_selected

    coordinates_list = 0

    if len(map_data) > 0:
        coordinates_list = map_data[0]['geometry']['coordinates']
        #coordinates = [coord for sublist in coordinates_list for coord in sublist]
        #coordinates_list = pd.DataFrame(coordinates, columns=['longitude', 'latitude'])
    else: 
        print("No Selected Region")
    coordinates_list
    return
"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"

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

        # Example: inject polygon info into context
        q = messages[-1]["content"]
        if polygon is not None:
            q += f"\nThe user has drawn a polygon with {len(polygon.exterior.coords)} points."
        return chain.invoke({"question": q})

    return mo.ui.chat(my_model)


@app.cell
def _(MapboxDrawControls, MapboxDrawOptions, leafmap):
    test = leafmap.Map(center=[0, 0], zoom=1, style="positron")
    draw_options = MapboxDrawOptions(
    display_controls_default=False,
    controls=MapboxDrawControls(polygon=True, line_string=False, point=True, trash=True),)
    test.add_draw_control(draw_options)
    test
    return (test,)


if __name__ == "__main__":
    app.run()
