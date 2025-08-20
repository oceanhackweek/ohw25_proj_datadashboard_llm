from ipyleaflet import Map, DrawControl, basemaps, Rectangle, LayersControl
import ipywidgets as w
from IPython.display import display, clear_output
from math import inf

def geometry_bbox(geo_json_geometry):
    def walk_coords(coords):
        if isinstance(coords[0], (float, int)):  
            yield coords
        else:
            for c in coords:
                yield from walk_coords(c)

    min_lon, min_lat = inf, inf
    max_lon, max_lat = -inf, -inf
    for lon, lat in walk_coords(geo_json_geometry["coordinates"]):
        min_lon, min_lat = min(min_lat, lat), min(min_lon, lon)  # swapped
        max_lon, max_lat = max(max_lat, lat), max(max_lon, lon)
    return [min_lon, min_lat, max_lon, max_lat]

def valid_bbox(b):
    min_lon, min_lat, max_lon, max_lat = b
    return (-180 <= min_lon < max_lon <= 180) and (-90 <= min_lat < max_lat <= 90)

m = Map(center=(20, 0), zoom=2, basemap=basemaps.OpenStreetMap.Mapnik)
m.layout.height = "520px"
m.add_control(LayersControl(position="topright"))

draw = DrawControl(
    rectangle={"shapeOptions": {"color": "#ff7800"}}, 
    polygon={}, circlemarker={}, circle={}, polyline={}, marker={}
)
m.add_control(draw)

rect_layer = None
bbox = None 

min_lat_w = w.FloatText(description="Min Lat")
min_lon_w = w.FloatText(description="Min Lon")
max_lat_w = w.FloatText(description="Max Lat")
max_lon_w = w.FloatText(description="Max Lon")

use_btn   = w.Button(description="Use these coordinates", button_style="success")
clear_btn = w.Button(description="Clear", button_style="warning")
status_out = w.Output()

def set_status(text):
    with status_out:
        clear_output()
        print(text)

def draw_rectangle_from_bbox(b):
    global rect_layer
    if rect_layer and rect_layer in m.layers:
        m.remove_layer(rect_layer)
    rect_layer = Rectangle(
        bounds=((b[1], b[0]), (b[3], b[2])),
        color="#ff7800",
        fill_color="#ff7800",
        fill_opacity=0.1,
    )
    m.add_layer(rect_layer)

def on_use_clicked(_):
    global bbox
    b = [min_lon_w.value, min_lat_w.value, max_lon_w.value, max_lat_w.value]
    if valid_bbox(b):
        bbox = b
        draw_rectangle_from_bbox(b)
        set_status(f"BBOX set: {bbox}")
    else:
        set_status("Invalid BBOX (check values).")

def on_clear_clicked(_):
    global bbox, rect_layer
    bbox = None
    if rect_layer and rect_layer in m.layers:
        m.remove_layer(rect_layer)
        rect_layer = None
    min_lat_w.value = min_lon_w.value = max_lat_w.value = max_lon_w.value = 0.0
    set_status("Cleared.")

use_btn.on_click(on_use_clicked)
clear_btn.on_click(on_clear_clicked)

def handle_draw(target, action, geo_json):
    global bbox
    g = geo_json["geometry"]
    b = geometry_bbox(g)
    min_lon_w.value, min_lat_w.value, max_lon_w.value, max_lat_w.value = b
    bbox = b
    set_status(f"ROI drawn → BBOX {bbox}")

draw.on_draw(handle_draw)

display(m)
display(
    w.VBox([
        w.HTML("<b>Study area</b>"),
        w.HBox([min_lat_w, min_lon_w, max_lat_w, max_lon_w]),
        w.HBox([use_btn, clear_btn]),
        status_out
    ])
)

