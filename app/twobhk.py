from dash import html
import dash_bootstrap_components as dbc
import warnings
from app import plan1_north, plan2_north
warnings.filterwarnings('ignore')

import pathlib

PATH = pathlib.Path(__file__).parent


layout = html.Div([html.H3('2 BHK', style={'textAlign': 'center'}),
            dbc.Row([
                dbc.Col(html.Div(plan1_north.layout), width=6),
                dbc.Col(html.Div(plan2_north.layout), width=6)]),
    ])
