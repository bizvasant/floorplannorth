import plotly.express as px
from dash import Input, Output, html, dcc, State
import cv2
import os
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from best_features import best_features
from main import app
import warnings
import base64
from contour import img_contour
from ocr import ocr

warnings.filterwarnings('ignore')
import pathlib

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
])

def parse_contents(contents,filename):
    # contents = str(contents[0])
    encoded_data = contents.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    fig = px.imshow(img)
    fig.update_yaxes(tick0=0, dtick=200)
    fig.update_xaxes(tick0=0, dtick=200)
    output = html.Div([dcc.Graph(figure=fig)])
    cv2.imwrite(filename, img)
    try:
        df1 = ocr.ocr_info(filename)
        # df1.to_csv("df1_image.csv", index=False)
        df = df1.sort_values(['Features'],ascending=True)
        df.to_csv("data_img_2.csv", index=False)
        # df = pd.read_csv("data_img_2.csv")
        df['Area in sq. ft.'] = df['Area in sq. ft.'].astype(str)
        df = df.replace('[\([{})\]]','',regex=True)
        table = dbc.Table.from_dataframe(df, bordered=True)
        output = html.Div([dcc.Graph(figure=fig), table])
        try:
            image = cv2.imread(filename)
            contour, th = img_contour.find_contour(image)
            x, y, w, h, cx, cy = img_contour.find_center(img, contour, th)
            qads = img_contour.draw_contour(img, x, y, w, h, cx, cy)
            fig_quad = px.imshow(qads)
            output = html.Div([dcc.Graph(figure=fig), table,
            dcc.Graph(figure=fig_quad)])
            try:
                from quadrants_area_1 import quadrants_area
                qaud_info = quadrants_area.quad_area_info(image,filename)
                qaud_info.to_csv("quads_img_2.csv", index=False)
                qaud_df = pd.read_csv("quads_img_2.csv")
                area_dist_df = qaud_df.groupby('Quadrant').Feature_Quadrant_area.sum().reset_index()
                area_dist_df.rename(columns = {'Feature_Quadrant_area':'Area distribution per quadrant'}, inplace = True)
                area_dist_df['Area distribution per quadrant'] = area_dist_df['Area distribution per quadrant'].astype(int)
                area_dist_df.rename(columns= {'Area distribution per quadrant':'Total Area of Quadrant (Sq. ft.)'},inplace = True)       
                area_dist = dbc.Table.from_dataframe(area_dist_df, bordered=True, style={'textAlign': 'center'})
                qaud_df.rename(columns = 
                {'Actual_area':'Feature Area (Sq. ft.)','Feature_Quadrant_area':'Feature Area in Quadrant (Sq. ft.)' },
                inplace = True)
                qaud_info = dbc.Table.from_dataframe(qaud_df, bordered=True)
                # fig_quad.update_yaxes(visible=False)
                # fig_quad.update_xaxes(visible=False)
                fig_quad.update_yaxes(tick0=0, dtick=200)
                fig_quad.update_xaxes(tick0=0, dtick=200)
                # from app import summary
                output = html.Div([dcc.Graph(figure=fig), table,
                        dcc.Graph(figure=fig_quad),area_dist,
                        qaud_info, summary_layout])
            except:
                output = html.Div([dcc.Graph(figure=fig), table,
                dcc.Graph(figure=fig_quad), html.H3("Unable to calculate the area of features for provided image.")])
        except:
            output = html.Div([dcc.Graph(figure=fig), table,
            html.H3("Quadrants are not drawn correctly for provided image.")])
    except:
        output = html.Div([dcc.Graph(figure=fig), html.H3("Unable to fetch textual information from provided image.")])
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


summary_layout = html.Div([html.Br(), dbc.Button("Summary", id="button", external_link=True,
                              style={"background-color": "#292929", "height": "40px"}), html.Br(), # NOQA E501
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

    return dbc.Container([html.H3("Plan Comparison", className="display-6",
            style={'textAlign': 'left'}),
    dbc.Table.from_dataframe(df_comp, bordered=True),
    html.H3("Conclusion", className="display-6",
            style={'textAlign': 'left'}),
    dbc.Card(html.P(conclusion), body=True)])