import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go

import etl.covid_data as covid
import etl.map_coordinate_data as map_coord
import etl.pop_data as pop

app = dash.Dash(__name__)
server = app.server

# covid.update_data()
# CONFIG
theme_color = {
    'background': '#111111',
    'text': '#BEBEBE',
    'grid': '#333333',
    'positive': '#5bc246',
    'negative': '#BF0000',
    'increasing': '#4f4fe0',
    'decreasing': '#db9d4b',
    'neutral': '#c2d690',
    'land': '#647d64',
    'ocean': '#9ecdff'
}
format_ind_number_neutral = {'valueformat': ',',
                             'font': {'size': 45,
                                      'color': theme_color['neutral']}}
format_ind_number_positive = {'valueformat': ',',
                              'font': {'size': 45,
                                       'color': theme_color['positive']}}
format_ind_number_negative = {'valueformat': ',',
                              'font': {'size': 45,
                                       'color': theme_color['negative']}}
format_ind_delta = {'increasing': {'color': theme_color['increasing']},
                    'decreasing': {'color': theme_color['decreasing']},
                    'font': {'size': 20}}
format_ind_title = {'font': {'size': 20,
                             'color': '#437d69'}}
# END CONFIG

# BASE DATAFRAMES
df_pop_clean = pop.get_fact_pop_clean()
world_geo_json = map_coord.get_world_geo_json()

df_covid_jhu_full = covid.get_fact_jhu_full()

df_covid_jhu_full_country = df_covid_jhu_full[~(df_covid_jhu_full['continent'].isna())]
df_covid_jhu_full_world = df_covid_jhu_full[df_covid_jhu_full['location'] == 'World']
df_covid_jhu_full_asia = df_covid_jhu_full[df_covid_jhu_full['location'] == 'Asia']
df_covid_jhu_full_eu = df_covid_jhu_full[df_covid_jhu_full['location'] == 'Europe']
df_covid_jhu_full_american = df_covid_jhu_full[df_covid_jhu_full['location'] == 'North America']
df_covid_jhu_full_americas = df_covid_jhu_full[df_covid_jhu_full['location'] == 'South America']
df_covid_jhu_full_ocean = df_covid_jhu_full[df_covid_jhu_full['location'] == 'Oceania']
df_covid_jhu_full_africa = df_covid_jhu_full[df_covid_jhu_full['location'] == 'Africa']

df_date = pd.DataFrame(df_covid_jhu_full_world.date.reset_index(drop=True))
df_date['date_label'] = df_date.apply(lambda d: {'style': {'transform': 'rotate(30deg) translate(0px, 7px)'},
                                                 'label': d.date.strftime('%d-%b-%Y')}
if (d.name < len(df_date) - 10) & (d.name % 10 == 0) else '', axis=1)

# END BASE DATAFRAMES

# Filtered dataframes
df_continent = pd.DataFrame()
df_country = pd.DataFrame()
scope = ''


# FILTER FUNCTIONS
# Filter continent - whole report
def filter_continent(selected_continent: str = 'World'):
    global df_continent, df_country, scope

    if selected_continent == 'World':
        df_continent = df_covid_jhu_full_world
        scope = 'world'
    elif selected_continent == 'Africa':
        df_continent = df_covid_jhu_full_africa
        scope = 'africa'
    elif selected_continent == 'Asia':
        df_continent = df_covid_jhu_full_asia
        scope = 'asia'
    elif selected_continent == 'Europe':
        df_continent = df_covid_jhu_full_eu
        scope = 'europe'
    elif selected_continent == 'North America':
        df_continent = df_covid_jhu_full_american
        scope = 'north america'
    elif selected_continent == 'Oceania':
        df_continent = df_covid_jhu_full_ocean
        scope = 'world'  # There is no scope for Oceania
    elif selected_continent == 'South America':
        df_continent = df_covid_jhu_full_americas
        scope = 'south america'

    if selected_continent == 'World':
        df_country = df_covid_jhu_full[~(df_covid_jhu_full['continent'].isna())]
    else:
        df_country = df_covid_jhu_full[df_covid_jhu_full['continent'] == selected_continent]


# Filter date
def filter_date(from_date: np.datetime64 = min(df_covid_jhu_full_country['date']),
                to_date: np.datetime64 = max(df_covid_jhu_full_country['date'])):
    global df_continent, df_country
    df_continent = df_continent[(df_continent['date'] >= from_date) & (df_continent['date'] <= to_date)]
    df_country = df_country[(df_country['date'] >= from_date) & (df_country['date'] <= to_date)]


# Filter country
def filter_country(selected_country: list = None):
    global df_country
    if selected_country is not None and len(selected_country) > 0:
        df_country = df_country[df_country['location'].isin(selected_country)]


# Initial filter continent (default filter)
filter_continent()

# Initial filter date (default filter)
filter_date()


# END FILTER FUNCTION


# CALL BACK FUNCTIONS
# INDICATORS
@app.callback(
    Output('fig_ind', 'figure'),
    [Input('filter_continent', 'value'),
     Input('filter_date', 'value'),
     Input('filter_country', 'value')])
def update_indicators(selected_continent, date_range, selected_country):
    filter_continent(selected_continent)
    filter_date(from_date=df_date.loc[date_range[0]:date_range[1]:1]['date'].to_list()[0],
                to_date=df_date.loc[date_range[0]:date_range[1]:1]['date'].to_list()[-1])
    filter_country(selected_country)

    fig_ind = go.Figure()
    # Death
    fig_ind.add_trace(go.Indicator(
        mode='number+delta',
        value=df_continent.death.iat[-1],
        delta={**{'reference': df_continent.death.iat[-2],
                  'relative': True,
                  'valueformat': '.2%'},
               **format_ind_delta},
        domain={'column': 0, 'row': 0, 'y': [0.99, 1]},
        number=format_ind_number_negative,
        title={**{'text': 'CUMULATIVE DEATHS'}, **format_ind_title}
    ))

    # Confirmed
    fig_ind.add_trace(go.Indicator(
        mode='number+delta',
        value=df_continent.confirm.iat[-1],
        delta={**{'reference': df_continent.confirm.iat[-2],
                  'relative': True,
                  'valueformat': '.2%'},
               **format_ind_delta},
        domain={'column': 1, 'row': 0, 'y': [0.99, 1]},
        number=format_ind_number_neutral,
        title={**{'text': 'UMULATIVE CONFIRMED CASES'}, **format_ind_title}
    ))

    # Recover
    fig_ind.add_trace(go.Indicator(
        mode='number+delta',
        value=df_continent.recover.iat[-1],
        delta={**{'reference': df_continent.recover.iat[-2],
                  'relative': True,
                  'valueformat': '.2%'},
               **format_ind_delta},
        domain={'column': 2, 'row': 0, 'y': [0.99, 1]},
        number=format_ind_number_positive,
        title={**{'text': 'CUMULATIVE RECOVERS'}, **format_ind_title}
    ))

    # New death
    fig_ind.add_trace(go.Indicator(mode='number',
                                   value=df_continent.new_death.iat[-1],
                                   domain={'column': 0, 'row': 1, 'y': [0, 0.01]},
                                   number=format_ind_number_negative,
                                   title={**{'text': 'NEW DEATHS IN DAY'}, **format_ind_title}
                                   ))

    # Active
    fig_ind.add_trace(go.Indicator(mode='number+delta',
                                   value=df_continent.active.iat[-1],
                                   delta={**{'reference': df_continent.active.iat[-2],
                                             'relative': False,
                                             'valueformat': ','},
                                          **format_ind_delta},
                                   domain={'column': 1, 'row': 1, 'y': [0, 0.01]},
                                   number=format_ind_number_neutral,
                                   title={**{'text': 'ACTIVE CASES'}, **format_ind_title}
                                   ))

    # New recover
    fig_ind.add_trace(go.Indicator(mode='number',
                                   value=df_continent.new_recover.iat[-1],
                                   domain={'column': 2, 'row': 1, 'y': [0, 0.01]},
                                   number=format_ind_number_positive,
                                   title={**{'text': 'NEW RECOVERS IN DAY'}, **format_ind_title}
                                   ))

    # Layout
    fig_ind.update_layout(grid={'rows': 2, 'columns': 3, 'pattern': 'independent'},
                          paper_bgcolor=theme_color['background'],
                          plot_bgcolor=theme_color['background'],
                          height=310
                          )
    # END INDICATORS
    return fig_ind


@app.callback(
    Output('fig_stacked_area', 'figure'),
    [Input('filter_continent', 'value'),
     Input('filter_date', 'value'),
     Input('filter_country', 'value')])
def update_stacked_area(selected_continent, date_range, selected_country):
    filter_continent(selected_continent)
    filter_date(from_date=df_date.loc[date_range[0]:date_range[1]:1]['date'].to_list()[0],
                to_date=df_date.loc[date_range[0]:date_range[1]:1]['date'].to_list()[-1])
    filter_country(selected_country)

    # STACKED AREA CHART
    fig_stacked_area = go.Figure()

    fig_stacked_area.add_trace(go.Scatter(x=df_continent['date'],
                                          y=df_continent['active'],
                                          mode='lines',
                                          line={'width': 1,
                                                'color': 'blue'},
                                          name='Active Cases',
                                          hoveron='points+fills',
                                          hoverinfo='text+x+y',
                                          hovertext='Active Cases',
                                          hovertemplate='%{y:.3s}',
                                          orientation='v',
                                          stackgroup='one',
                                          fill='tonexty'))
    fig_stacked_area.add_trace(go.Scatter(x=df_continent['date'],
                                          y=df_continent['recover'],
                                          mode='lines',
                                          line={'width': 1,
                                                'color': 'green'},
                                          name='Recovers',
                                          hoveron='points+fills',
                                          hoverinfo='text+x+y',
                                          hovertext='Recover',
                                          hovertemplate='%{y:.3s}',
                                          orientation='v',
                                          stackgroup='one',
                                          fill='tonexty'))
    fig_stacked_area.add_trace(go.Scatter(x=df_continent['date'],
                                          y=df_continent['death'],
                                          mode='lines',
                                          line={'width': 1,
                                                'color': 'red'},
                                          name='Deaths',
                                          hoveron='points+fills',
                                          hoverinfo='text+x+y',
                                          hovertext='Death',
                                          hovertemplate='%{y:.3s}',
                                          orientation='v',
                                          stackgroup='one',
                                          fill='tonexty'))
    fig_stacked_area.add_trace(go.Scatter(x=df_continent['date'],
                                          y=df_continent['confirm'],
                                          mode='lines',
                                          line={'width': 0,
                                                'color': '#a1a1a1'},
                                          name='Confirm',
                                          hoveron='points+fills',
                                          hoverinfo='text+x+y',
                                          hovertext='Confirm',
                                          hovertemplate='%{y:.3s}',
                                          orientation='v',
                                          stackgroup='',
                                          showlegend=False))

    fig_stacked_area.update_layout(
        xaxis={'gridcolor': theme_color['grid']},
        yaxis={'gridcolor': theme_color['grid']},
        hovermode='x',
        legend={'title': 'Metrics'},
        font_color=theme_color['text'],
        paper_bgcolor=theme_color['background'],
        plot_bgcolor=theme_color['background']
    )
    # END STACKED AREA CHART
    return fig_stacked_area


@app.callback(
    Output('fig_line', 'figure'),
    [Input('filter_continent', 'value'),
     Input('filter_date', 'value'),
     Input('filter_country', 'value')])
def update_line_chart(selected_continent, date_range, selected_country):
    filter_continent(selected_continent)
    filter_date(from_date=df_date.loc[date_range[0]:date_range[1]:1]['date'].to_list()[0],
                to_date=df_date.loc[date_range[0]:date_range[1]:1]['date'].to_list()[-1])
    filter_country(selected_country)

    # LINE CHART
    fig_line = go.Figure()

    df_country_latest = df_country[df_country.date == df_country.date.max()].sort_values(by='confirm', ascending=False,
                                                                                         axis=0)
    countries = df_country_latest['location'][0:10:1].to_list()

    for country in countries:
        df_line = df_country[df_country['location'] == country]
        fig_line.add_trace(go.Scatter(x=df_line['date'],
                                      y=df_line['active'],
                                      mode='lines',
                                      line={'width': 1},
                                      name=country,
                                      hoveron='points+fills',
                                      hoverinfo='text+x+y',
                                      hovertext='Active Cases',
                                      hovertemplate='%{y:.3s}',
                                      orientation='v'))

    fig_line.update_layout(
        xaxis={'gridcolor': theme_color['grid']},
        yaxis={'gridcolor': theme_color['grid']},
        hovermode='x',
        legend={'title': 'Countries'},
        font_color=theme_color['text'],
        paper_bgcolor=theme_color['background'],
        plot_bgcolor=theme_color['background']
    )
    # END LINE CHART
    return fig_line


@app.callback(
    Output('fig_map', 'figure'),
    [Input('filter_continent', 'value'),
     Input('filter_date', 'value'),
     Input('filter_country', 'value')])
def update_map(selected_continent, date_range, selected_country):
    filter_continent(selected_continent)
    filter_date(from_date=df_date.loc[date_range[0]:date_range[1]:1]['date'].to_list()[0],
                to_date=df_date.loc[date_range[0]:date_range[1]:1]['date'].to_list()[-1])
    filter_country(selected_country)

    # MAP 1
    df = df_country[df_country.date == df_country.date.max()]
    fig_map = go.Figure()
    fig_map.add_trace(go.Choropleth(
        locationmode='ISO-3',
        locations=df['iso_code'],
        z=df['confirm'],
        colorscale=['#7595ff', '#f1fc1c', '#fc1c1c'],
        colorbar={
            'title': 'Number of confirm cases',
            'thicknessmode': 'fraction',
            'lenmode': 'fraction',
            'thickness': 0.02,
            'len': 1,
            'tickmode': 'auto',
            'tickfont': {'color': theme_color['text']},
            'titlefont': {'color': theme_color['text']}
        },
        text=df['location'] + '<br>Confirmed: ' + df['confirm'].apply('{:,.0f}'.format)
             + '<br>Active: ' + df['active'].apply('{:,.0f}'.format),
        hoverinfo='text',
        hoverlabel={'bgcolor': '#3dffcf'}
    ))
    fig_map.add_trace(go.Scattergeo(
        locationmode='ISO-3',
        locations=df['iso_code'],
        marker={'size': df['active'],
                'sizeref': max(df['active']) / 3e6 * 3e4,
                'color': '#3dffcf',
                'opacity': 0.8},
        name='',
        text=df['location'] + '<br>Confirmed: ' + df['confirm'].apply('{:,.0f}'.format)
             + '<br>Active: ' + df['active'].apply('{:,.0f}'.format),
        hoverinfo='text'
    ))
    fig_map.update_geos(projection_type='natural earth',
                        scope=scope,
                        # landcolor=theme_color['land'],
                        oceancolor=theme_color['ocean'],
                        lakecolor=theme_color['ocean'],
                        showland=True,
                        showocean=True,
                        showcountries=True,
                        showlakes=True,
                        resolution=110  # 50 is better but so lag
                        )
    fig_map.update_layout(height=550,
                          title={'text': 'Number of Cumulative Confirmed Cases (color bar)<br>'
                                         'and Active Cases on ' + max(df['date']).strftime('%d-%b-%Y')
                                         + ' (size of markers)',
                                 'font': {'size': 15,
                                          'color': theme_color['text']}},
                          margin={"r": 0, "t": 75, "l": 0, "b": 0},
                          paper_bgcolor=theme_color['background'],
                          plot_bgcolor=theme_color['background']
                          )
    # END MAP 1
    return fig_map


@app.callback(
    Output('filter_country', 'options'),
    [Input('filter_continent', 'value')])
def update_filter_country(selected_continent):
    filter_continent(selected_continent)
    return [{'label': i, 'value': i} for i in df_country['location'].drop_duplicates()]


# @app.callback(
#     Output('filter_country', 'value'),
#     [Input('filter_continent', 'value')])
# def update_filter_country(selected_continent):
#     filter_continent(selected_continent)
#
#     return df_country['location'].drop_duplicates().to_list()

# END CALL BACK FUNCTIONS


# SCATTER PLOT ANIMATED
fig_scatter = go.Figure()
# END SCATTER PLOT ANIMATED

# PAGE LAYOUT
app.layout = \
    html.Div(children=[
        html.Div(dcc.RadioItems(id='filter_continent',
                                options=[{'label': i, 'value': i} for i in
                                         ['World', 'Africa', 'Asia', 'Europe', 'North America', 'Oceania',
                                          'South America']],
                                value='World',
                                labelStyle={'float': 'center', 'display': 'inline-block'}),
                 style={'textAlign': 'center',
                        'color': theme_color['text'],
                        'width': '100%',
                        'height': 50,
                        'float': 'center',
                        'display': 'inline-block'}),
        html.Div(dcc.Dropdown(id='filter_country',
                              multi=True,
                              style={'width': '90%',
                                     'float': 'center',
                                     'marginLeft': '5%'})
                 ),
        html.Div(dcc.RangeSlider(id='filter_date',
                                 min=0,
                                 max=(df_continent.date.max() - df_continent.date.min()).days,
                                 step=None,
                                 value=[0, (df_continent.date.max() - df_continent.date.min()).days],
                                 marks=df_date['date_label'].to_dict(),
                                 updatemode='mouseup'
                                 ),
                 style={'textAlign': 'center',
                        'color': theme_color['text'],
                        'width': '90%',
                        'marginLeft': '5%',
                        'height': 50,
                        'float': 'center',
                        'display': 'inline-block'}
                 ),
        html.Div(dcc.Graph(id='fig_ind'),
                 style=dict(textAlign='center',
                            width='90%',
                            marginLeft='5%',
                            display='inline-block')
                 ),
        html.Div(dcc.Graph(id='fig_stacked_area'),
                 style=dict(textAlign='center',
                            width='45%',
                            marginLeft='5%',
                            float='center',
                            display='inline-block')
                 ),
        html.Div(dcc.Graph(id='fig_line'),
                 style=dict(textAlign='center',
                            width='45%',
                            marginRight='5%',
                            float='center',
                            display='inline-block')
                 ),
        html.Div(dcc.Graph(id='fig_map'),
                 style=dict(textAlign='center',
                            width='90%',
                            marginLeft='5%',
                            float='center',
                            display='inline-block')
                 )
    ])

# Execute web app
if __name__ == '__main__':
    app.run_server(port=8844, debug=True)
