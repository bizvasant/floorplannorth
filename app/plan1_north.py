import plotly.express as px
from dash import Input, Output, html, dcc, State, no_update
import cv2
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from main import app
import warnings
import base64
import math
import os
import imutils
from math import pi as PI
from ocr import ocr
import matplotlib.pyplot as plt
from PIL import Image
from contour import img_contour
import pathlib
warnings.filterwarnings('ignore')

PATH = pathlib.Path(__file__).parent

layout = dbc.Container([
    html.Br(),
    dcc.Upload(id='upload-image_p1',
               children=html.Div(['Drag and Drop or ',
                                  html.A('Select Files')]),
               style={
                   'borderStyle': 'dashed',
                   'borderRadius': '5px',
                   'textAlign': 'center',
                   "height": "60px"
               },
               multiple=True),
    html.Br(),
    html.Div(id='output-image-upload_p1'),
    html.Pre(id="annotations-data_1")
])


def parse_contents(contents, filename):
    # contents = str(contents[0])
    encoded_data = contents.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    fig = px.imshow(img)
    fig.update_layout(dragmode="drawline",
                      newshape=dict(fillcolor="cyan",
                                    drawdirection="diagonal",
                                    opacity=0.5,
                                    line=dict(color="blue", width=2)))
    cv2.imwrite("img_1.png", img)
    try:
        df1 = ocr.ocr_info("img_1.png")
        df = df1.sort_values(['Features'], ascending=True)
        df['Area in sq. ft.'] = df['Area in sq. ft.'].astype(str)
        df = df.replace('[\([{})\]]', '', regex=True)
        df.to_csv("data_img_1.csv", index=False)
        table = dbc.Table.from_dataframe(df, bordered=True)
        output = html.Div([dcc.Graph(figure=fig, id="graph-picture_1"), table])
    except:
        output = html.Div([
            dcc.Graph(figure=fig, id="graph-picture_1"),
            html.H3("Unable to fetch textual information from provided image.")
        ])
    return html.Div([output])


@app.callback(Output('output-image-upload_p1', 'children'),
              Input('upload-image_p1', 'contents'),
              State('upload-image_p1', 'filename'))
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n)
            for c, n in zip(list_of_contents, list_of_names)
        ]
        return children


def image_rotate(image, angle):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel = image[1, 1]
    image_1 = imutils.rotate_bound(image, angle)
    new = Image.new(mode="RGB",
                    size=(image_1.shape[1], image_1.shape[0]),
                    color=tuple(pixel))
    image = new + imutils.rotate_bound(image, angle)

    contour, th = img_contour.find_contour(image)
    x, y, w, h, cx, cy = img_contour.find_center(image, contour, th)
    quads = img_contour.draw_contour(image, x, y, w, h, cx, cy)
    im_rgb = cv2.cvtColor(quads, cv2.COLOR_BGR2RGB)

    q2 = im_rgb[y:cy, x:cx]
    q3 = im_rgb[cy:y + (h), x:cx]
    q4 = im_rgb[cy:y + (h), cx:x + (w)]
    q1 = im_rgb[y:cy, cx:x + (w)]
    images = {
        "Quadrant2": q2,
        "Quadrant1": q1,
        "Quadrant3": q3,
        "Quadrant4": q4
    }

    fig = plt.figure(figsize=(10, 7))
    ax = []
    rows = 2
    columns = 2
    keys = list(images.keys())
    values = list(images.values())
    for i in range(rows * columns):
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title(keys[i])
        plt.imshow(images[keys[i]])

    plt.tight_layout()
    plt.savefig('quadplots_1.png', dpi=300)


def get_angle_degree(pt_dict_1, pt_dict_2):
    # Derive points
    pt1 = [pt_dict_1['x0'], pt_dict_1['y0']]
    pt2 = [pt_dict_1['x1'], pt_dict_1['y1']]
    pt3 = [pt_dict_2['x0'], pt_dict_2['y0']]
    pt4 = [pt_dict_2['x1'], pt_dict_2['y1']]
    # lines
    l1 = [pt1, pt2]
    l2 = [pt3, pt4]
    m1 = (l1[1][1] - l1[0][1]) / (l1[1][0] - l1[0][0])
    m2 = (l2[1][1] - l2[0][1]) / (l2[1][0] - l2[0][0])
    angle_rad = abs(math.atan(m1) - math.atan(m2))
    angle_deg = angle_rad * 180 / PI
    angle_deg = round(angle_deg, 2)
    return angle_deg


@app.callback(
    Output("annotations-data_1", "children"),
    Input("graph-picture_1", "relayoutData"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data):
    if "shapes" in relayout_data:
        relayout_values = list(relayout_data.values())
        relayout_list = list(relayout_values[0])
        if len(relayout_list) == 2:
            pt_dict_1 = dict(relayout_list[0])
            pt_dict_2 = dict(relayout_list[1])
            angle_deg = get_angle_degree(pt_dict_1, pt_dict_2)
            img = cv2.imread("img_1.png")
            var = image_rotate(img, angle_deg)
            quad_img = cv2.imread('quadplots_1.png')
            fig = px.imshow(quad_img)
            fig.update_yaxes(visible=False)
            fig.update_xaxes(visible=False)
            return html.Div([dcc.Graph(figure=fig)])
        else:
            no_update
    else:
        return no_update