import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    import folium
    import leafmap.maplibregl as leafmap
    #import leafmap.foliumap as leafmap
    from maplibre.plugins import MapboxDrawControls, MapboxDrawOptions
    return MapboxDrawControls, MapboxDrawOptions, leafmap, mo


@app.cell
def _(mo):
    mo.md("<h1>Data Dashboard with Chatbot!").center()
    return


@app.cell
def _(mo):
    mo.md("<h0> OceanHackWeek 2025").center()
    return


@app.cell(hide_code=True)
def _(MapboxDrawControls, MapboxDrawOptions, leafmap):
    test = leafmap.Map(center=[0, 0], zoom=1, style="positron")
    draw_options = MapboxDrawOptions(
        display_controls_default=False,
        controls=MapboxDrawControls(polygon=True, line_string=False, point=True, trash=True),
    )
    test.add_draw_control(draw_options)
    test
    return (test,)


@app.cell
def _(test):
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


@app.cell
def _(context, mo, query_llm):
    def my_model(messages, config):
        question = messages[-1].content
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        # Query your own model or third-party models
        response = query_llm(prompt, config)
        return response

    mo.ui.chat(my_model)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
