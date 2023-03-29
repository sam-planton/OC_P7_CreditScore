import pandas as pd
import streamlit as st
import numpy as np
import shap
import mlflow
import matplotlib.pyplot as plt
import matplotlib
from streamlit_shap import st_shap
import plotly.express as px
import plotly.figure_factory as ff
import requests
import plotly.graph_objects as go
import time
import os.path as op
from streamlit.components.v1 import html


@st.cache_data
def fetch_data():
    # To lighten the load in memory, set columns to float32 directly when loading
    # (load a sample of the dataset to identify columns)

    nrows_train = 19950
    nrows_test = 50

    # Load the training data
    sample_df = pd.read_csv('data/Xtrain.csv', nrows=50, index_col='SK_ID_CURR')
    float_cols = [c for c in sample_df if sample_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    Xtrain = pd.read_csv('data/Xtrain.csv', engine='c', dtype=float32_cols, nrows=nrows_train, index_col='SK_ID_CURR')
    Xtrain_addinfo = pd.read_csv('data/Xtrain_addinfo.csv', nrows=nrows_train, index_col='SK_ID_CURR')

    # Load the testing data (sample clients)
    sample_df = pd.read_csv('data/Xtest.csv', nrows=50, index_col='SK_ID_CURR')
    float_cols = [c for c in sample_df if sample_df[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    Xtest = pd.read_csv('data/Xtest.csv', engine='c', dtype=float32_cols, nrows=nrows_test, index_col='SK_ID_CURR')
    Xtest_addinfo = pd.read_csv('data/Xtest_addinfo.csv', nrows=nrows_train, index_col='SK_ID_CURR')

    # Load features description
    description_df = pd.read_csv('data/description_df.csv')

    return Xtrain, Xtrain_addinfo, Xtest, Xtest_addinfo, description_df


@st.cache_resource
def fetch_model():
    # Load the model saved with MLflow
    model = mlflow.sklearn.load_model('data/model')

    return model


@st.cache_resource
def compute_shap(_model, Xtrain, N_shap_samples):
    # How many samples to use (more samples takes time!)
    Nsamples = N_shap_samples

    # Compute SHAP values using training data & model
    # get only predict of 2nd class
    def predict_fn(x): return _model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, shap.kmeans(Xtrain, 20))
    shap_values = explainer.shap_values(Xtrain.sample(Nsamples), gc_collect=True)

    return explainer, shap_values


def show_client_data(data_client, data_group, description_df, varlist, do_subgrp):
    theme_plotly = None  # None or 'streamlit'

    data_group.loc[data_group['model_predict_class'] == 0.0, 'CreditStatus'] = 'Crédit accordé'
    data_group.loc[data_group['model_predict_class'] == 1.0, 'CreditStatus'] = 'Crédit refusé'
    cols = st.columns(2)
    for i in range(len(cols)):
        with cols[i]:
            options = ['<Sélectionnez une variable>'] + varlist
            var = st.selectbox(f'Sélectionnez la variable à visualiser', options, key=i)
            if var != options[0]:

                # === Distribution plot, for continuous variables
                if data_group[var].nunique() > 2:
                    if do_subgrp:
                        data1 = data_group.loc[data_group['model_predict_class'] == 0.0, var]
                        data2 = data_group.loc[data_group['model_predict_class'] == 1.0, var]
                        hist_data = [data1, data2]
                        group_labels = ['Crédit accordé', 'Crédit refusé']
                        colors = ['forestgreen', 'red']
                    else:
                        hist_data = [data_group.loc[:, var]]
                        group_labels = ['Tous statuts']
                        colors = ['darkorange']
                    fig = ff.create_distplot(hist_data, group_labels, colors=colors, show_hist=False, show_rug=False,
                                             curve_type="kde", histnorm="probability")
                    # Add client line
                    fig.add_vline(x=data_client[var][0], line_dash='solid', line_color='yellow', line_width=3,
                                  annotation=dict(text="Client", font_size=18, showarrow=True, arrowhead=1, ax=0,
                                                  ay=-20, arrowcolor='white'), annotation_position='top')
                    # Set layout
                    fig.update_traces(line=dict(width=3))  # only if show_hist=False
                    fig.update_layout(xaxis=dict(tickfont=dict(size=18)))
                    fig.update_layout(legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='left', x=0, ), )
                    fig.update_yaxes(showticklabels=False, showgrid=False, title='Densité', title_font=dict(size=18),
                                     tickfont=dict(size=18))

                # === Bar plot, for binary variables
                elif data_group[var].nunique() <= 2:
                    client_val = data_client[var][0]

                    # Specific labels for one variable
                    if var == 'CODE_GENDER':
                        val0 = 'F '
                        val1 = 'M '
                    else:
                        val0 = 'Non '
                        val1 = 'Oui '

                    if do_subgrp:
                        counts_df = pd.DataFrame(data_group.value_counts(subset=['CreditStatus', var]),
                                                 columns=['count'])
                        counts_df = counts_df.reset_index()
                        fig = px.bar(counts_df, x='count', y=var, color='CreditStatus', barmode='group',
                                     orientation='h', color_discrete_sequence=['forestgreen', 'red'])
                        # Add client arrow
                        fig.add_annotation(x=np.max(counts_df.loc[counts_df[var] == client_val, 'count']), y=client_val,
                                           ax=40, ay=0, text="Client", font_size=18, showarrow=True, arrowhead=1,
                                           arrowcolor='white')
                    else:
                        counts = data_group[var].value_counts()
                        counts_df = pd.DataFrame({'value': counts.index, 'count': counts.values})
                        fig = px.bar(counts_df, x='count', y='value', orientation='h',
                                     color_discrete_sequence=['darkorange'])
                        # Add client arrow
                        fig.add_annotation(x=counts_df.loc[counts_df['value'] == client_val, 'count'].iloc[0],
                                           y=client_val, ax=40, ay=0, text="Client", font_size=18, showarrow=True,
                                           arrowhead=1, arrowcolor='white')
                    # Set layout
                    fig.update_layout(
                        legend=dict(orientation='h', yanchor='top', y=1.15, xanchor='left', x=0, title=''))
                    fig.update_layout(yaxis=dict(tickmode='array', ticktext=[val0, val1], tickvals=[0, 1]))
                    fig.update_layout(xaxis=dict(tickfont=dict(size=18)), yaxis=dict(tickfont=dict(size=18)))
                    fig.update_xaxes(title='Nombre de clients')
                    fig.update_yaxes(showticklabels=True, showgrid=False, title='')

                # Show plot
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

                # Show info about var and group
                description = description_df.loc[description_df['Row_name'] == var, 'Description'].values[0]
                client_val = data_client[var][0]
                group_mean = data_group[var].mean()
                group_std = data_group[var].std()
                group_median = data_group[var].median()
                rows = [
                    f'**Description**: {description}',
                    f'**Client**: {client_val:.02f}',
                    f'**Moyenne groupe**: {group_mean:.02f} (+/- {group_std:.02f})',
                    f'**Médiane groupe**: {group_median:.02f}'
                ]
                st.write('  \n'.join(rows))


def show_client_data_bivariate(data_client, data_group, varlist):
    col1, col2, col3 = st.columns([1, 1, 1])
    options = ['<Sélectionnez la variable 1>'] + varlist

    with col1:
        var1 = st.selectbox(f'Sélectionnez la variable 1', options, key=5551)
    with col2:
        var2 = st.selectbox(f'Sélectionnez la variable 2', options, key=5552)
    with col3:
        # grphtype = 'Heatmap'
        grphtype = st.radio("Type de graphique :", ("Heatmap", "Scatterplot"))
    if (var1 != options[0]) & (var2 != options[0]):
        var3 = 'Score'
        # We add jitter for binary (categorical) variables
        jitter_amount = 0.06  # adjust the amount of jitter as needed
        data_group_jitter = data_group[:500].copy()  # make a copy of the data
        if len(np.unique(data_group[var1])) == 2:
            data_group_jitter[var1] += np.random.normal(0, jitter_amount, len(data_group_jitter))
        if len(np.unique(data_group[var2])) == 2:
            data_group_jitter[var2] += np.random.normal(0, jitter_amount, len(data_group_jitter))
        data_group_jitter = data_group_jitter.round(2)
        if grphtype == 'Scatterplot':
            # With sccaterplot
            fig = px.scatter(data_group_jitter, x=var1, y=var2, color=var3, color_continuous_scale='RdYlGn')
        elif grphtype == 'Heatmap':
            # With heatmap
            fig = px.density_contour(data_group_jitter, x=var1, y=var2, z=var3, histfunc="avg")
            fig.update_traces(contours_coloring="fill", contours_showlabels=False, colorscale='RdYlGn')
            fig.update_traces(colorbar_title='Score Moyen', selector=dict(type='histogram2dcontour'))
            # fig = px.density_heatmap(data_group, x=var1, y=var2, z=var3, histfunc="avg",
            #                          color_continuous_scale='RdYlGn', nbinsx=100, nbinsy=100)
        # Add client dot
        fig.add_scatter(x=[data_client[var1].values[0]], y=[data_client[var2].values[0]], mode='markers', name='Client',
                        marker=dict(size=18, color='yellow', line=dict(width=2, color='black')))
        # Move the legend for the yellow dot above the colorbar
        fig.update_layout(legend=dict(yanchor="top", y=1.08, xanchor="left", x=0.01))
        # Layout
        fig.update_coloraxes(colorbar_title_side='right')
        fig.update_layout(height=650, width=650)
        st.plotly_chart(fig, use_container_width=False, theme=None)


def gauge_animated_figure(score, score_threshold, frame_duration=0.1):
    # Define the initial sscore value
    sscore = 0

    # Create the initial Plotly figure with the sscore value
    gaugefig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sscore,
        title={'text': "Score crédit"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "yellow"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, score_threshold], 'color': 'red'},
                {'range': [score_threshold, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "yellow", 'width': 4},
                'thickness': 0.75,
                'value': sscore
            }
        }
    ))
    gaugefig.update_layout(height=350)

    # Create the Streamlit plotly_chart object with the initial Plotly figure
    plotly_chart = st.plotly_chart(gaugefig, use_container_width=True)

    # Update the sscore value in the loop and update the Plotly figure and Streamlit plotly_chart object
    for sscore in range(1, int(score), 1):
        gaugefig.update_traces(
            value=sscore,
            gauge={
                'threshold': {
                    'value': sscore
                }
            }
        )
        plotly_chart.plotly_chart(gaugefig, use_container_width=True)
        time.sleep(frame_duration)

    # Final: real score
    gaugefig.update_traces(
        value=score,
        gauge={
            'threshold': {
                'value': score
            }
        }
    )
    plotly_chart.plotly_chart(gaugefig, use_container_width=True)


def main():
    # ================================= SETUP ================================= #
    # Streamlit page config
    st.set_page_config(page_title="Credit Score", page_icon="🏦", layout="wide")

    # Flask API endpoint to return model prediction
    API_endpoint = "https://sp-oc-p7-api.herokuapp.com/predict"

    # =========================== FETCH DATA & MODEL =========================== #
    Xtrain, Xtrain_addinfo, Xtest, Xtest_addinfo, description_df = fetch_data()
    model = fetch_model()
    metrics = pd.read_csv(op.join('data', 'run_info.csv'))

    # Compute feature importances based on the SHAP values
    N_shap_samples = 500
    explainer, shap_values = compute_shap(model, Xtrain, N_shap_samples)
    feature_importances_df = pd.DataFrame(np.mean(np.abs(shap_values), axis=0), index=Xtrain.columns, columns=['SHAP'])

    # =========================== DASHBOARD ITEMS =========================== #
    # Title
    st.title('🏦 Prediction score crédit 💵')

    # Style
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Info
    st.markdown("""
                Cette application permet d'attribuer un **"score crédit"**
                 à un client basé sur la probabilité de défaut du client,
                déterminée à partir d'un modèle de machine learning.  
                Les données utilisées proviennent du jeu de données Kaggle
                [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)
                
                Code et données accessibles sur [GitHub](https://github.com/sam-planton/OC_P7_CreditScore)                
                """, unsafe_allow_html=True)

    with st.expander("Cliquez ici pour développer les instructions"):
        st.write("""
                Pour commencer, veuillez choisir un des clients dans le menu de la barre latérale. Il s'agit 
                d'une liste de client dont les informations sont connues et dont on souhaite déterminer le
                *score crédit*.   
                
                Veuillez ensuite sélectionner un des quatre onglets ci-dessous.
                - **🗠 Visualiser les données du client**: Cet onglet vous permet de visualiser les données du 
                groupe de clients enregistrés dans la base de données, tout en y situant le client d'intérêt,
                sur une ou plusieurs variables ou choix (plusieurs centaines disponibles).  
                
                - **💯 Calculer le score crédit du client**: Cet onglet utilise le modèle de machine learning 
                entraîné, mis à disposition sous la forme d'une API, pour prédire le score du client sélectionné.
                Le score varie de 0 à 100. Un score de 0 indique une probabilité de défaut du client maximale.    
                
                - **📊 Explications du score**: Les informations fournies sous cet onglet permettent de mieux 
                comprendre le score prédit par le modèle pour le client sélectionné : quelles sont les variables
                qui font augmenter ou diminuer son score et en quelles proportions.
                
                - **⚙️Informations sur le modèle**: Cet onglet fournit des informations sur les performances
                du modèle statistique de prédiction, ainsi que sur les variables les plus importantes pour la 
                détermination du *score crédit*. 
                """)

    tab1, tab2, tab3, tab4 = st.tabs(['🗠 Visualiser les données du client',
                                      '💯 Calculer le score crédit du client',
                                      '📊 Explications du score',
                                      '⚙️ Informations sur le modèle'])

    # ======= Sidebar select client ======= #
    clist = ['Client ' + str(x + 1) for x in Xtest.index]
    clist = ['<Sélectionnez un client>'] + clist
    client_id = clist[0]  # default
    score = 0
    client_id = st.sidebar.selectbox('Sélectionnez le client (ou entrez son identifiant)', clist, key=1000)
    if client_id == clist[0]:
        pass
    else:

        # Client data
        data_sample = Xtest.iloc[[clist.index(client_id) - 1]]
        data_sample_addinfo = Xtest_addinfo.iloc[[clist.index(client_id) - 1]]
        data_sample = data_sample.reset_index(drop=True)
        gender = data_sample['CODE_GENDER'].apply(lambda x: 'M' if x == 1.0 else 'F')

        # Display basic client data
        st.sidebar.subheader('Données de base')
        st.sidebar.metric('Genre', '%s' % gender.values[0])
        st.sidebar.metric('Age', '%d ans' % np.ceil(-1 * data_sample['DAYS_BIRTH'] / 365.25))
        st.sidebar.metric('Revenus', '{:,}$'.format(int(data_sample['AMT_INCOME_TOTAL'].values[0])).replace(',', ' '))
        st.sidebar.metric("Nombre d'enfants", '%d' % data_sample['CNT_CHILDREN'])

    # ======= Show client data ======= #
    with tab1:
        st.header('🗠 Visualiser les données du client')
        # st.write(''' Cet onglet vous permet de visualiser les données du groupe de clients enregistrés dans la base de
        # données, tout en y situant le client d'intérêt.''')
        if client_id == clist[0]:
            st.write('_Veuillez sélectionner un des clients dans le menu de la barre latérale_')
        else:
            # Concatenate clients from train and test dataset
            Xgroup1 = Xtrain.join(Xtrain_addinfo)
            Xgroup2 = Xtest.join(Xtest_addinfo)
            Xgroup = pd.concat([Xgroup1, Xgroup2])
            Xgroup['Score'] = np.round((1 - Xgroup['model_predict_proba']) * 100, 0)

            # Visualisation options
            st.write('')
            st.write('')
            st.subheader('Paramètres')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                options = ['Tous les clients',
                           'Même catégorie socio-démographique',
                           'Même groupe - proche sur toutes variables']
                helptxt = 'Comparer à tous les clients  \n' \
                          'Au groupe de clients similaires sur un sous-ensemble de variables socio-démographiques  \n' \
                          'Au groupe de clients similaires sur une combinaisons de toutes les variables'
                option = st.selectbox(f'Sélectionnez la base de comparaison client', options, help=helptxt)
                if option == options[0]:
                    Xgroup = Xgroup
                elif option == options[1]:
                    mask = Xgroup['cluster_SocioDemo'] == data_sample_addinfo['cluster_SocioDemo'].values[0]
                    Xgroup = Xgroup[mask]
                elif option == options[2]:
                    mask = Xgroup['cluster_FullData'] == data_sample_addinfo['cluster_FullData'].values[0]
                    Xgroup = Xgroup[mask]
                st.write('%d clients sélectionnés dans la base de données' % len(Xgroup))
            with col3:
                helptxt = 'Toutes, ou limiter aux 15 plus importantes selon le modèle'
                genre = st.radio("Quelles variables rechercher ?",
                                 ('Toutes', 'Uniquement les plus importantes pour le score'), help=helptxt)
                if genre == 'Toutes':
                    varlist = sorted(data_sample.columns)
                elif genre == 'Uniquement les plus importantes pour le score':
                    feature_importances_df = feature_importances_df.sort_values(by='SHAP', ascending=False)
                    varlist = sorted(feature_importances_df[:15].index)
            with col1:
                helptxt = 'Séparer ou non les données du groupe selon la prediction du modèle: crédit accepté ou refusé.'
                subgrp = st.radio("Sous-groupes selon la prédiction",
                                  ('Regrouper tous les clients', 'Séparer selon la prédiction du modèle'), help=helptxt)
                if subgrp == 'Regrouper tous les clients':
                    do_subgrp = False
                elif subgrp == 'Séparer selon la prédiction du modèle':
                    do_subgrp = True

            # Generate one-var figures
            st.write('')
            st.write('')
            st.subheader('Graphiques par variable unique')
            st.write('''Vous pouvez ici visualiser la distribution des données pour une variable au choix.
                     La position du client d'intérêt est indiquée.''')
            show_client_data(data_sample, Xgroup, description_df, varlist, do_subgrp)

            # Generate bivariate figure
            st.write('')
            st.write('')
            st.subheader('Graphique par paire de variables')
            st.write('''Vous pouvez ici visualiser la valeur des clients du groupe, sur deux variables au choix
                     simultanément.  
                     La couleur correspond au score prédit pour les clients du groupe.  
                     La position du client d'intérêt est affichée par un point jaune.''')
            show_client_data_bivariate(data_sample, Xgroup, varlist)

    # ======= Show client score prediction by API call  ======= #
    with tab2:
        st.header('💯 Calculer le score crédit du client')
        if client_id == clist[0]:
            st.write('_Veuillez sélectionner un des clients dans le menu de la barre latérale_')
        else:
            col1, col2 = st.columns([0.5, 1])
            with col1:
                st.write('     \n')
                predict_btn = st.button('Calcul du score', type='primary')

                threshold = metrics['metrics.cv_test_best_threshold'][0]
                score_threshold = (1 - threshold) * 100
                txt = ('Seuil de décision  \n  (score minimal, recommandé = %d) :' % score_threshold)
                score_threshold = st.slider(txt, min_value=0, max_value=100, value=int(score_threshold), step=1,
                                            format='%d')

            if predict_btn:

                # === Get prediction and score
                # using API on the server
                st.write('Model API: ' + API_endpoint)
                response = requests.post(API_endpoint, json=data_sample.to_dict())
                pred = response.json()[0]

                # # using downloaded model
                # pred = model.predict_proba(data_sample)[0][1]
                score = (1 - pred) * 100

                col1, col2 = st.columns([0.5, 1])
                with col1:

                    # === Gauge figure with progress animation
                    gauge_animated_figure(score, score_threshold, frame_duration=0.05)

                    # === st.metric
                    with col2:
                        subcol1, subcol2 = st.columns([0.5, 1])
                        with subcol1:
                            st.metric('Score crédit:', value=('%d/100' % score))
                            st.write('Probabilité de ne pas rembourser : %.01f%%' % (pred * 100))
                            st.sidebar.metric('Score :', value=('%d/100' % score))

                        # === Message
                        with subcol2:
                            if score < score_threshold:
                                st.markdown(
                                    '<h1 style="text-align:center;color:red;font-weight:700;font-size:26px">Risque de défaut élevé.  \n\nRecommandation : refuser le crédit.</h1>',
                                    unsafe_allow_html=True)
                            elif score >= score_threshold:
                                st.markdown(
                                    '<h1 style="text-align:center;color:green;font-weight:700;font-size:26px">Risque de défaut faible.  \n\nRecommandation : accorder le crédit.</h1>',
                                    unsafe_allow_html=True)
                                st.balloons()

    # ======= Score explanations  ======= #
    with tab3:
        score = 50
        st.header('📊 Explications du score')
        if client_id == clist[0]:
            st.write(
                '_Veuillez sélectionner un des clients dans le menu de la barre latérale_')
        else:
            if score == 0:
                st.write('_Veuillez calculer le score du client_')
            else:
                # Compute SHAP values for the client
                shap_values_client = explainer.shap_values(data_sample)

                # Shap values for the sample client in a df
                shap_sample_df = pd.DataFrame(shap_values_client[0], index=data_sample.columns, columns=['SHAP'])
                shap_sample_df['asb_SHAP'] = shap_sample_df['SHAP'].abs()
                shap_sample_df = shap_sample_df.sort_values(by='asb_SHAP', ascending=False)
                # Add descriptions
                shap_sample_df = shap_sample_df.merge(description_df, left_index=True, right_on='Row_name', how='left')

                # Display features descriptions:
                col1, col2 = st.columns(2)
                with col1:
                    st.write('Les variables suivantes font augmenter le score de ce client '
                             '👍📈 (diminuer la probabilité de non-remboursement) :')
                    shap_negat_df = shap_sample_df.sort_values(by='SHAP', ascending=True).iloc[:10]
                    st.write(shap_negat_df[['Row_name', 'SHAP', 'Description']])
                with col2:
                    st.write('Les variables suivantes font diminuer le score de ce client '
                             '👎📉 (augmenter la probabilité de non-remboursement) :')
                    shap_posit_df = shap_sample_df.sort_values(by='SHAP', ascending=False).iloc[:10]
                    st.write(shap_posit_df[['Row_name', 'SHAP', 'Description']])  # nicer but shows index...
                # Colors
                negative_color = "#228B22"  # green
                positive_color = "#e50000"  # red

                # Force plot
                st.subheader('Force plot')
                st.write(
                    'Les variables en vert font baisser la probabilité de défaut (augmenter le score), '
                    'les variables en rouge la font augmenter (diminuer le score). '
                    'La taille de chaque barre est proportionnelle à l''impact de la variable sur le score.')
                # fig, ax = plt.subplots(facecolor='None')

                fig = shap.force_plot(explainer.expected_value, shap_values_client,
                                      data_sample, plot_cmap=[positive_color, negative_color])
                with st.container():
                    st_shap(fig)  # Display the plot in the Streamlit app
                # st_shap(fig, width=1200)  # Display the plot in the Streamlit app

                # Waterfall_plot
                st.subheader('Waterfall plot')
                st.write(
                    'Les variables en vert font baisser la probabilité de défaut (augmenter le score), '
                    'les variables en rouge la font augmenter (diminuer le score). '
                    'La taille de chaque barre est proportionnelle à l''impact de la variable sur le score.')
                fig, ax = plt.subplots(facecolor='None')
                ax.set_facecolor("None")
                shap.waterfall_plot(shap.Explanation(values=shap_values_client[0],
                                                     base_values=explainer.expected_value,
                                                     data=data_sample.iloc[0]),
                                    max_display=15, show=False)
                # Change the colormap of the artists
                # Default SHAP colors
                default_pos_color = "#ff0051"
                default_neg_color = "#008bfb"
                for fc in plt.gcf().get_children():
                    for fcc in fc.get_children():
                        if isinstance(fcc, matplotlib.patches.FancyArrow):
                            if matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color:
                                fcc.set_facecolor(positive_color)
                            elif matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color:
                                fcc.set_color(negative_color)
                        elif isinstance(fcc, plt.Text):
                            if matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color:
                                fcc.set_color(positive_color)
                            elif matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color:
                                fcc.set_color(negative_color)
                plt.setp(ax.get_xticklabels(), color="white")
                plt.setp(ax.get_yticklabels(), color="white")
                for spine in ax.spines.values():
                    spine.set_color('white')
                with st.container():
                    st_shap(fig, height=700)
                # st_shap(fig, height=700, width=1000)

    # ======= Show model info  ======= #
    with tab4:
        st.header('⚙️Informations sur le modèle')

        # Metrics
        st.subheader('Métriques')
        cols = st.columns(6)
        with cols[0]:
            st.metric('Score métier', np.round(metrics['metrics.cv_test_custom_score'][0], 2))
        with cols[1]:
            st.metric('ROC AUC', np.round(metrics['metrics.cv_test_roc_auc'][0], 2))
        with cols[2]:
            st.metric('F1 score', np.round(metrics['metrics.cv_test_f1'][0], 2))
        with cols[3]:
            st.metric('Accuracy', np.round(metrics['metrics.cv_test_accuracy'][0], 2))

        # Summary plots
        matplotlib.rcParams['text.color'] = 'white'
        st.subheader('Importance des features')
        st.write('Une valeur élevée implique une une forte importance de la variable dans '
                 'le calcul des prédictions du modèle')

        top_features = feature_importances_df.sort_values(by='SHAP', ascending=False)[:15]
        fig = px.bar(top_features, x='SHAP', y=top_features.index, orientation='h',
                     color_discrete_sequence=['forestgreen'])
        fig.update_xaxes(title_text='mean(|SHAP value|)<br>Impact moyen sur la prédiction du modèle', tickfont_size=16,
                         title_font_size=20)
        fig.update_yaxes(title_text='', tickfont_size=16)
        fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True, theme='streamlit')

        st.subheader('Distribution de la contribution des features')
        st.write('''Les couleurs chaudes désignent une valeur élevée sur la feature, les couleurs froides une valeur basse.  
        Par exemple, les points verts avec une valeur SHAP positive (à droite) désignent des clients pour qui une augmentation de la valeur
        sur la feature implique une augmentation de la probabilité de non-remboursement.  A l'opposé, les points verts avec une valeur SHAP négative
        (à gauche) désignent des clients pour qui une augmentation de la valeur sur la feature implique une diminution de la probabilité de non-remboursement.''')
        fig, ax = plt.subplots(facecolor='None')
        im = shap.summary_plot(shap_values, features=Xtrain[:N_shap_samples],
                          plot_type='dot', max_display=15, cmap="RdYlGn", alpha=1, axis_color='w', show=False)
        # Set colors to white
        ax.set_facecolor("None")
        plt.setp(ax.get_xticklabels(), color="white", fontsize=10)
        plt.setp(ax.get_yticklabels(), color="white", fontsize=10)
        for spine in ax.spines.values():
            spine.set_color('white')
        fig.axes[1].tick_params(colors='white')  # colorbar
        st_shap(fig, height=600, width=1000)

        st.write('')
        st.write('')
        st.subheader('Analyse du data drift')
        st.write('''Cette analyse réalisée en utilisant un jeu de données de test vise à évaluer la stabilité
                    des données au cours du temps. Le jeu de données de test étant utilisé pour simuler une
                    possible évolution future du jeu de données d'entraînement.''')
        html_file = open("data/evidently_Report_DataDriftPreset.html", 'r', encoding='utf-8').read()
        html(html_file, height=800, scrolling=True)

    st.markdown("---")


if __name__ == '__main__':
    main()
