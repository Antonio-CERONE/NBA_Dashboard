#!/usr/bin/env python
# coding: utf-8

# # Get some data and plot them into plotly

# In[1]:


import plotly.express as px
import plotly.graph_objects as go # Import de graph_objects de plotly pour avoir plus de souplesse au lieu de ployly express
import pandas as pd
import numpy as np
from datetime import datetime


# In[2]:


# Data put together by Gabe Salzer on data.world
# Data source: http://www.landofbasketball.com/nba_teams_year_by_year.htm
df = pd.read_csv("https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Plotly_Graphs/Heatmaps/Historical%20NBA%20Performance.csv")


# In[3]:


df.info()


# In[4]:


df.head(10)


# In[5]:


df = pd.pivot_table(df,
                    index = "Team",
                    columns = "Year",
                    values = "Winning Percentage"
             )
df


# In[6]:


df.columns


# In[7]:


df_T = df.T
df_T
print(df_T.info())


# In[8]:


df_T.head()


# In[9]:


# Years restriction
list_restriction = range(2000,2016,1)
print (list_restriction)
print(len(list_restriction))

df_T = df_T[df_T.index.isin(list_restriction)]
df_T.head(10)


# In[10]:


df= df_T.T


# In[11]:


# Teams restriction
df = df[df.index.isin(['Warriors',  'Celtics', 'Bulls','Supersonics'])]


# In[12]:


#Plotly graph : heatmap
fig = px.imshow(df, 
                color_continuous_scale=px.colors.sequential.YlOrBr,
                title="NBA Season Winning Percentage")
fig.update_layout(title_font={'size':27}, 
                  title_x=0.5)
fig.update_traces(hoverongaps=False,
                  )
fig.show()


# # Using Dash with the data

# In[13]:


import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
from datetime import datetime


# In[14]:


# Charger les données
df = pd.read_csv("https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Plotly_Graphs/Heatmaps/Historical%20NBA%20Performance.csv")
df_p = pd.pivot_table(
    df,
    index="Team",
    columns="Year",
    values="Winning Percentage",
)
print(df_p.shape)


# In[15]:


df_p.head()


# In[16]:


# Définir la liste des équipes
list_equipe=df_p.index.sort_values(ascending=False).to_list()
print(type(list_equipe))
print(list_equipe[0])
print(list_equipe)


# In[17]:


# Définir la liste des années
list_years = df["Year"].unique().tolist()

# Définir les dates min et max
min_year = min(list_years)
max_year = max(list_years)

# Initialiser les dates par défaut pour le DatePickerRange
initial_start_date = datetime(min_year, 1, 1)
initial_end_date = datetime(max_year, 12, 31)


# In[18]:


# Initialiser l'application Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# In[19]:


# Créer le contenu Markdown
markdown_content = """
## Objectif                          
Ici nous allons analyser le **pourcentage de victoire des équipes de basket américaines de la NBA**.
Cela va être possible sur une assez **longue période de temps**.

### Interactivité - Lisibilité
Afin d'assurer un bon engagement de l'utilisateur, voici deux éléments importants d'interactivité
- Il sera possible de <u>*choisir les équipes à comparer*</u> : 
    - Conserver un nombre assez restreint d'équipes choisies, cela facilitera les comparaisons
- Il sera aussi possible de <u>*restreindre la plage de temps*</u> de l'analyse
    - _**The less, the best**_
                                   
Pour *accroître la qualité de l'expérience utilisateur*, nous offrons aussi le choix du thème des couleurs pour l'affichage des graphiques.
"""


# In[20]:


# Créer le layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Pourcentage de Victoire - NBA"),
        ], width=12)
    ]),

    html.Br(),
    html.Br(),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    html.Label([
                        "Choix du thème de l'affichage des données",
                        
                        html.Br(),
                        html.Br(),
                        
                        dcc.Dropdown(
                            id="colorscale-dropdown",
                            clearable=False,
                            value="plasma",
                            options=[{"label": color, "value": color} for color in px.colors.named_colorscales()],
                            style={'width': '100%'}  # Définir la largeur du dropdown
                        )
                    ])
                ], width=4),  # Largeur de 4 pour le premier dropdown
                
                dbc.Col([
                    html.Label([
                        "Choix des équipes de baskets",
                        
                        html.Br(),
                        html.Br(),
                        
                        dcc.Dropdown(
                            id="team_dropdown",
                            clearable=False,
                            value=["Spurs"],
                            options=[{"label": nom_equipe, "value": nom_equipe} for nom_equipe in list_equipe],
                            multi=True,
                            style={'width': '100%'}  # Définir la largeur du dropdown 
                        )
                    ])
                ], width=4),  # Largeur de 4 pour le deuxième dropdown
                
                dbc.Col([
                    html.Label([
                        "Choix de la plage de temps",
                        
                        html.Br(),
                        html.Br(),
                        
                        dcc.DatePickerRange(
                            id="date-picker-range",
                            start_date=initial_start_date,
                            end_date=initial_end_date,
                            min_date_allowed=initial_start_date,
                            max_date_allowed=initial_end_date,
                            display_format='YYYY',
                            style={'width': '100%', 'display': 'inline-block'}
                        )
                    ])
                ], width=4),  # Largeur de 4 pour le troisième dropdown
            ]),
            
            html.Br(),
            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Afficher/Masquer Description",
                        id="collapse-button",
                        color="primary",
                        n_clicks=0
                    )
                ], width=12),
                dbc.Col([
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody(
                                dcc.Markdown(
                                    children=markdown_content,
                                    dangerously_allow_html=True
                                )
                            ),
                            style={'margin-top': '10px'}
                        ),
                        id="collapse",
                        is_open=True  # Par défaut, la section est ouverte
                    )
                ], width=12)
            ])
        ], width=12)
    ]),

    html.Br(),
    html.Br(),

    dbc.Row([
        dbc.Col([
            html.H3("Heatmap des victoires NBA"),
            dcc.Graph(id="heatmap", figure={})
        ], width=12)
    ]),

    html.Br(),
    html.Br(),

    dbc.Row([
        dbc.Col([
            html.H3("Évolution du Taux de Victoire"),
            dcc.Graph(id="line-plot", figure={})
        ], width=12)
    ])
])


# In[21]:


# Callback pour gérer le dépliage/réduction de la section Markdown
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")]
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Callback pour mettre à jour le heatmap et le line plot
@app.callback(
    [Output(component_id="heatmap", component_property='figure'),
     Output(component_id="line-plot", component_property='figure')],
    [
        Input(component_id="colorscale-dropdown", component_property='value'),
        Input(component_id="team_dropdown", component_property='value'),
        Input(component_id="date-picker-range", component_property='start_date'),
        Input(component_id="date-picker-range", component_property='end_date'),
    ]
)
def update_figures(colorscale_value, selected_teams, start_date, end_date):
    # Filtrer les données en fonction des équipes sélectionnées
    if not selected_teams:
        selected_teams = list_equipe  # Si aucune équipe n'est sélectionnée, utiliser toutes les équipes par défaut
    
    filtered_df = df_p.loc[selected_teams]
    
    # Convertir les dates en années
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    
    # Filtrer les données en fonction de la plage de temps sélectionnée
    filtered_df = filtered_df.loc[:, start_year:end_year]
    
    # Vérifier si le DataFrame filtré est vide
    if filtered_df.empty:
        heatmap_fig = {
            "data": [],
            "layout": go.Layout(
                title="Aucune donnée disponible pour les critères sélectionnés",
                xaxis_title="Année",
                yaxis_title="Équipe",
                height=800
            )
        }
        line_plot_fig = {
            "data": [],
            "layout": go.Layout(
                title="Aucune donnée disponible pour les critères sélectionnés",
                xaxis_title="Année",
                yaxis_title="Taux de Victoire",
                height=600
            )
        }
        return heatmap_fig, line_plot_fig
    
    # Création du heatmap avec Plotly Graph Objects
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=filtered_df.values,
            x=filtered_df.columns,
            y=filtered_df.index,
            colorscale=colorscale_value,
            colorbar=dict(title="Taux de Victoire"),
            hovertemplate=( "<b>Année</b>: %{x}<br>" + "<b>Équipe</b>: %{y}<br>" + "<b>Taux de Victoire</b>: %{z:.1%}<extra></extra>" )
        )
    )

    heatmap_fig.update_layout(
        title="Heatmap des victoires NBA",
        xaxis_title="Année",
        yaxis_title="Équipe",
        height=800
    )

    # Création du line plot avec Plotly Graph Objects
    if 2 <= len(selected_teams) <= 8 and (end_year - start_year + 1) >= 6:
        line_plot_fig = go.Figure()
        x_values = filtered_df.columns.astype(int)
        
        # Calculer la moyenne par année pour toutes les équipes sélectionnées
        mean_by_year = filtered_df.mean(axis=0)
        
        # Ajouter la moyenne par année en ligne pointillée rouge épaisse
        line_plot_fig.add_trace(go.Scatter(
            x=x_values,
            y=mean_by_year,
            mode='lines',
            name='Moyenne par année',
            line=dict(color='red', dash='dash', width=3)
        ))
        
        # Lissage des données par moyenne mobile de 3 ans
        for team in selected_teams:
            y_values = filtered_df.loc[team].values
            y_smoothed = pd.Series(y_values).rolling(window=3, center=True, min_periods=1).mean().values
            line_plot_fig.add_trace(go.Scatter(x=x_values, y=y_smoothed, mode='lines', name=team))
        
        line_plot_fig.update_layout(
            title="Évolution du Taux de Victoire",
            xaxis_title="Année",
            yaxis_title="Taux de Victoire",
            height=600
        )
    else:
        line_plot_fig = {
            "data": [],
            "layout": go.Layout(
                title="Veuillez sélectionner entre 2 et 8 équipes et une plage de temps d'au moins 6 ans",
                xaxis_title="Année",
                yaxis_title="Taux de Victoire",
                height=600
            )
        }

    return heatmap_fig, line_plot_fig


# In[22]:


# Exécuter l'application
if __name__ == '__main__':
    # app.run_server(debug=True, port=8007) # If running  within an IDE but not within Notebook
    app.run_server(debug=False, port=8007) # If running  within Jupyter Notebook but not within an IDE


# In[ ]:




