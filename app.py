from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import os
from main import app
from app import onebhk, threebhk, twobhk

app.layout = html.Div(
    [dcc.Location(id='url', refresh=True),
     html.Div(id='page-content')])

app.title = "Bizmetric Floor Plan System"

index_page = dbc.Container([html.H1('Floor Plan Usecase', style={'textAlign': 'center'}),html.Br(),
                           dbc.DropdownMenu([
        dbc.DropdownMenuItem("1 BHK", href='/app/onebhk',style={"width": "1100px"}),
        dbc.DropdownMenuItem("2 BHK", href='/app/twobhk',style={"width": "1100px"}),
        dbc.DropdownMenuItem("3 BHK", href='/app/threebhk',style={"width": "1100px"}),
    ], label="Select Flat", toggle_style={"width": "1100px"})])


# Update the index
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname=index_page):
    if pathname == '/app/onebhk':
        return html.Div([index_page, html.Br(), onebhk.layout])
    elif pathname == '/app/twobhk':
        return html.Div([index_page, html.Br(), twobhk.layout])
    elif pathname == '/app/threebhk':
        return html.Div([index_page, html.Br(), threebhk.layout])
    else:
        return index_page

if __name__ == '__main__':
    # app.run_server(debug=False, use_reloader=False, port=7920)
    app.run_server(host='0.0.0.0', port=80, debug=False, use_reloader=False)