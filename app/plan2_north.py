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
from best_features import best_features


PATH = pathlib.Path(__file__).parent


layout = dbc.Container([
    html.Br(),
    dcc.Upload(id='upload-image_p2',
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
    html.Div(id='output-image-upload_p2'),
    html.Pre(id="annotations-data_2")
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
    cv2.imwrite("img_2.png", img)
    try:
        df1 = ocr.ocr_info("img_2.png")
        df = df1.sort_values(['Features'], ascending=True)
        df['Area in sq. ft.'] = df['Area in sq. ft.'].astype(str)
        df = df.replace('[\([{})\]]', '', regex=True)
        df.to_csv("data_img_2.csv", index=False)
        table = dbc.Table.from_dataframe(df, bordered=True)
        output = html.Div([dcc.Graph(figure=fig, id="graph-picture_2"), table])
    except:
        output = html.Div([
            dcc.Graph(figure=fig),
            html.H3("Unable to fetch textual information from provided image.")
        ])
    return html.Div([output])


@app.callback(Output('output-image-upload_p2', 'children'),
              Input('upload-image_p2', 'contents'),
              State('upload-image_p2', 'filename'))
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
    plt.savefig('quadplots_2.png', dpi=300)


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
            # For quadrants distribution
    if (pt_dict_2['x1'] > pt_dict_2['x0'] ) & (pt_dict_2['y1'] > pt_dict_2['y0']):
        print("Q2")
        angle_deg = 180 - abs(angle_deg)
    elif (pt_dict_2['x1'] > pt_dict_2['x0'] ) & (pt_dict_2['y1'] < pt_dict_2['y0']):
        print("Q3")
        angle_deg = 180 + abs(angle_deg)
    elif (pt_dict_2['x1'] < pt_dict_2['x0'] ) & (pt_dict_2['y1'] < pt_dict_2['y0']):
        print("Q4")
        angle_deg = 360 - abs(angle_deg)
    print(angle_deg)
    return angle_deg


@app.callback(
    Output("annotations-data_2", "children"),
    Input("graph-picture_2", "relayoutData"),
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
            img = cv2.imread("img_2.png")
            var = image_rotate(img, angle_deg)
            quad_img = cv2.imread('quadplots_2.png')
            fig = px.imshow(quad_img)
            fig.update_yaxes(visible=False)
            fig.update_xaxes(visible=False)
            return html.Div([dcc.Graph(figure=fig), summary_layout])
        else:
            no_update
    else:
        return no_update


summary_layout = html.Div([html.Br(), dbc.Button("Summary", id="button", external_link=True,
                              style={"background-color": "#292929", "height": "40px"}), # NOQA E501
                   html.Div(id="summary")])  # NOQA E501

@app.callback(
    Output("summary", "children"),
    Input("button", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    df1 = pd.read_csv("data_img_1.csv")
    df2 = pd.read_csv("data_img_2.csv")

    # data_1 = best_features.pre_process(df1)
    # data_2 = best_features.pre_process(df2)

    print("dataframed are read")

    output_file = best_features.best_feature(df1,df2)

    print("got best features")
    df = output_file.head(20)

    df.to_csv("best_features.csv",index= False)
    feature_df = pd.read_csv("best_features.csv")

    if feature_df["floorplan_1"].sum() > feature_df["floorplan_2"].sum():
        conclusion = "Floor plan 1 is better than Floor plan 2."
    if feature_df["floorplan_1"].sum() < feature_df["floorplan_2"].sum():
        conclusion = "Floor plan 2 is better than Floor plan 1."
    if feature_df["floorplan_1"].sum() == feature_df["floorplan_2"].sum():
        conclusion = "Floor plan 1 and Floor plan 2 both are same."

    summary, df_comp = best_features.comp(df)

    try:
        ext = ('.png', '.jpg', '.csv')
        for file in os.listdir():
            if file.endswith(ext):
                print("Removing ", file)
                os.remove(file)
    except Exception as e: print(e)
    finally:
        print("No garbage available")

    return dbc.Container([html.Br(), html.H3("Plan Comparison", className="display-6",
            style={'textAlign': 'left'}),
    dbc.Table.from_dataframe(df_comp, bordered=True),
    html.H3("Conclusion", className="display-6",
            style={'textAlign': 'left'}),
    dbc.Card(html.P(conclusion), body=True)])