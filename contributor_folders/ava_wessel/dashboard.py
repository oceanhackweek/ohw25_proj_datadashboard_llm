import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    import geopandas as gpd
    import openlayers as ol

    from langchain_community.vectorstores import Chroma
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import ConversationalRetrievalChain
    from langchain.schema import Document
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    return ChatOpenAI, ChatPromptTemplate, StrOutputParser, mo, ol


@app.cell
def _(mo):
    mo.md(
        """
    <div style="display: flex; align-items: center; justify-content: center; gap: 12px;">
        <img src="https://oceanhackweek.org/_static/logo.png" 
             alt="Logo" width="50" height="50" style="border-radius: 0%;">
        <h1 style="font-family: Arial, sans-serif; font-size: 36px; color: #004080; margin: 0; font-weight: bold">
            Data Dashboard with Chatbot!
        </h1>
    </div>

    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    <h1 style = "font-family:Arial, sans-serif; font-size:15px;color: #00408;  font-weight: bold"> OceanHackWeek 2025
    """).center()
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
    return (map_frame,)


@app.cell
def _():
    return


@app.cell
def _(
    ChatOpenAI,
    ChatPromptTemplate,
    StrOutputParser,
    map_boundaries,
    map_frame,
    mo,
):
    def my_model(messages, config):
        HF_TOKEN = "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"

        def map_boundaries(area):
            if len(map_frame) > 0:
                return print(map_frame)

        llm = ChatOpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
            model="openai/gpt-oss-20b:fireworks-ai"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI assistant specializing in ocean data. "
                       "You have access to the map boundaries from the UI map."
                       "that the user can draw."),
            ("human", "{question}")
        ])

        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({"question": "{question}"},
                               {'map boundaries': ""})



        return response

    mo.ui.chat(my_model, map_boundaries(map_frame))
    return


if __name__ == "__main__":
    app.run()
