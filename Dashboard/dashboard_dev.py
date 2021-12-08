# ------------ Libraries import ------------
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import requests
import pickle
from PIL import Image
# ------------ Data Import -----------------
## Import passenger train data ##
df_test_sample = pd.read_csv(r'test_sample_data_home_risk.csv', index_col=0)
### last feature is a probability (will be used for visualization purpose) ###
features = df_test_sample.columns[: -1] 
loan_id = df_test_sample.index.sort_values()
treshold = 0.49
## import the illustration image ##
img = Image.open(r'logo_projet_fintech.png')
## loading the standard scaler to display normal values of client features ##
std_scaler = pickle.load(open(r'std_scaler_home_risk.pkl', 'rb'))

# ------------ Sidebar configuration -------
## add side bar for user to interact ##

### Display the image with streamlit ###
st.sidebar.image(img)
### Add column for user input ###
st.sidebar.header('Sélectionner une demande de prêt:')
selected_credit = st.sidebar.selectbox('Prêt_ID', loan_id)
### Add checkbox for displaying different client informations ###
client_data = st.sidebar.checkbox('Données client')
client_pred_score = st.sidebar.checkbox('Résultat de la demande de prêt')

# ------------ Main display, part by part ---
## Generic title ##
st.write('# **SENECHAL Yannick: Projet 7 "Prêt à dépenser" / Formation OpenClassRooms DataScientist**')
st.write("## **Classification d'une demande de crédit**")

## Display input dataframe with multiselection of features for all the passenger list available ##
st.write('### Informations générales clients (index = ID de la demande de prêt):')
st.write('Dimension des données: ' + str(df_test_sample[features].shape[0]) + ' lignes ' + str(df_test_sample[features].shape[1]) + ' colonnes')
selections = st.multiselect('Vous pouvez ajouter ou enlever une feature présente dans cette liste:', df_test_sample[features].columns.tolist(),
 df_test_sample[features].columns.tolist()[0:4])
st.dataframe(df_test_sample.loc[:,selections])

## Display selected client data (checkbox condition: 'Données client') ##
if client_data:
    st.write(f'### Données du client, demande {selected_credit}')
    ### define values to display for the barchart and client data (with a maximum at 5) ###
    selections_client0 = st.multiselect('Vous pouvez afficher 5 features maximum parmi cette liste:', df_test_sample[features].columns.tolist(),
    df_test_sample[features].columns.tolist()[0:2])
    ### define columns to split some visual in two ###
    col1, col2 = st.beta_columns(2)
    ### Display client informations regarding selected features ###
    col1.dataframe(df_test_sample[features].loc[selected_credit, selections_client0])
    ### define pyplot for col2 barchart with selected passenger informations with condition of the number of selected features ###
    if len(selections_client0) <= 5:
        fig_client_info = plt.figure()
        plt.title(f'Diagramme bar données ID: {selected_credit}')
        sns.barplot(x=df_test_sample[features].loc[selected_credit, selections_client0].index, y=df_test_sample[features].loc[selected_credit, selections_client0].values)
        plt.xlabel('Features')
        plt.ylabel('Valeur')
        #### Display the graph ####
        col2.pyplot(fig_client_info)
    else:
        col2.write("Vous avez sélectionné trop de feature!!! Le graphique n'est pas affiché")

## Display loan answer regarding model probability calcul (path through API Flask to get the result / checbox condition : 'Résultat de la demande de prêt') ##
if client_pred_score:
    st.write('### Demande de crédit sélectionnée: Décision')
    ### careful the url of the API should be change for serial deployment!! ###
    url_api_model_result = 'http://127.0.0.1:5000/scores'
    ### Be careful to the params, with must have a dict with index / ID loan value. It is how it is implemented in our API ###
    get_request = requests.get(url=url_api_model_result, params={'index': selected_credit})
    ### We get  the prediction information from the json format of the API model ###
    prediction_value = get_request.json()['Credit_score']
    ### We get the answer regardin loan accpetation ###
    answer_value = bool(get_request.json()['Answer'])
    ### Display results ###
    st.write(f'Demande de crédit ID: {selected_credit}')
    st.write(f'Probabilité de défauts de remboursement: {prediction_value*100:.2f} %')
    if answer_value:
        st.write('Demande de crédit acceptée!')
    else:
        #### add condition in function of the value of the prediction, if over the treshold but near should be discussed ####
        if prediction_value > treshold and prediction_value <= 0.55:
            st.write('Demande de crédit refusée --> à discuter avec le conseiller')
        else:
            st.write('Demande de crédit refusée!')
    ### add gauge for the prediction value with plotly library ###
    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = float(f'{prediction_value*100:.1f}'),
    mode = "gauge+number+delta",
    title = {'text': "Score(%)"},
    delta = {'reference': treshold*100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
    gauge = {'axis': {'range': [0, 100]},
             'bar': {'color': 'black'},
             'steps' : [
                 {'range': [0, 30], 'color': "darkgreen"},
                 {'range': [30, (treshold*100)], 'color': "lightgreen"},
                 {'range': [(treshold*100),55], 'color': "orange"},
                 {'range': [55, 100], 'color':"red"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': treshold*100}}))
    st.plotly_chart(fig_gauge)

