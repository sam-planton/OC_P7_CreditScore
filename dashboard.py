import pandas as pd
import streamlit as st
import requests
import numpy as np
import shap
import mlflow
import matplotlib.pyplot as plt
import matplotlib
from streamlit_shap import st_shap
import plotly.express as px
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects
import os


@st.cache_data
def fetch_data(artifacts_uri):
    # Load the training data
    # art_uri = 'runs:/'+run_id+'/Xtrain.csv'
    art_uri = f"{artifacts_uri}/Xtrain.csv"
    Xtrain = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri),
                         index_col='SK_ID_CURR')

    # art_uri = 'runs:/'+run_id+'/Xtrain_clusters.csv'
    art_uri = f"{artifacts_uri}/Xtrain_clusters.csv"
    Xtrain_clusters = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri),
                                  index_col='SK_ID_CURR')

    # Load sample clients
    # art_uri = 'runs:/'+run_id+'/Xtest_samples.csv'
    art_uri = f"{artifacts_uri}/Xtest_samples.csv"
    Xtest_samples = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri),
                                index_col='SK_ID_CURR')
    # art_uri = 'runs:/'+run_id+'/Xtest_samples_clusters.csv'
    art_uri = f"{artifacts_uri}/Xtest_samples_clusters.csv"
    Xtest_samples_clusters = pd.read_csv(
        mlflow.artifacts.download_artifacts(art_uri), index_col='SK_ID_CURR')

    # Load features description
    # desc_uri = 'runs:/'+run_id+'/description_df.csv'
    art_uri = f"{artifacts_uri}/description_df.csv"
    description_df = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri))

    # # Load some metrics
    # run = mlflow.get_run(run_id)
    # Optimal_threshold = run.data.metrics['Optimal_threshold']
    # roc_auc =  run.data.metrics['test_roc_auc_cv']
    # accuracy =  run.data.metrics['test_accuracy_cv']
    # custom_score_v2 =  run.data.metrics['test_custom_score_v2_cv']
    # metrics = {'Optimal_threshold':Optimal_threshold,
    #            'Accuracy':accuracy,
    #            'ROC AUC':roc_auc,
    #            'Score métier':custom_score_v2}

    return Xtrain, Xtrain_clusters, Xtest_samples, Xtest_samples_clusters, description_df


@st.cache_resource
def fetch_model(artifacts_uri):
    # Load the saved model from MLflow
    model_uri = f"{artifacts_uri}/model"
    model = mlflow.sklearn.load_model(model_uri)

    return model


@st.cache_resource
def compute_shap(_model, Xtrain):
    # Compute SHAP values using training data & model
    def predict_fn(x): return _model.predict_proba(x)[:,
                              1]  # to get only predict of 2nd class

    explainer = shap.KernelExplainer(predict_fn, shap.kmeans(Xtrain, 30))
    shap_values = explainer.shap_values(Xtrain.sample(50))

    return explainer, shap_values


def color_gauge(score, threshold):
    score = (1 - score) * 100
    threshold = (1 - threshold) * 100
    min_val = 0
    max_val = 100

    fig, ax = plt.subplots(figsize=(8, 0.5), facecolor='None')
    cmap = plt.get_cmap('RdYlGn')
    norm = plt.Normalize(min_val, max_val)

    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)

    rect = FancyBboxPatch((0, 0), score, 100,
                          boxstyle='round, pad=1, rounding_size=5',
                          linewidth=0,
                          color=cmap(norm(score)),
                          clip_on=False,
                          alpha=0.9,
                          zorder=2,
                          antialiased=True,
                          snap=True,
                          path_effects=[path_effects.withSimplePatchShadow()])
    ax.add_patch(rect)
    ax.add_patch(rect)
    ax.axvline(x=threshold, color='yellow', linestyle='-', linewidth=2)
    ax.annotate('Seuil', xy=(threshold, 100), xytext=(threshold, 180),
                ha='center', color='white', fontsize=10,
                arrowprops=dict(facecolor='white', arrowstyle='-|>',
                                color='white'),
                )
    ax.set_facecolor("None")
    plt.setp(ax.get_xticklabels(), color="white", fontsize=10)
    plt.setp(ax.get_yticklabels(), color="white", fontsize=10)
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.yticks([])

    return fig


def show_client_data(data_client, data_group, description_df, varlist):
    theme_plotly = None  # None or streamlit

    cols = st.columns(3)
    for i in range(len(cols)):
        with cols[i]:
            options = ['<Sélectionnez une variable>'] + varlist
            value = options[0]  # default
            var = st.selectbox(f'Sélectionnez la variable à visualiser',
                               options, key=i)
            if var != options[0]:
                if data_group[var].nunique() > 2:
                    fig = px.histogram(data_group, x=var, log_x=False,
                                       histnorm="density",
                                       color_discrete_sequence=[
                                           'forestgreen'])  # nbins=60,
                    fig.add_vline(x=data_client[var][0], line_dash='solid',
                                  line_color='red', line_width=3,
                                  annotation=dict(text="Client", font_size=18,
                                                  showarrow=True, arrowhead=1,
                                                  ax=0, ay=-20,
                                                  arrowcolor='white'),
                                  annotation_position='top')
                    fig.update_layout(xaxis=dict(tickfont=dict(size=18)))
                    # fig.update_layout(showlegend=False, yticklabels=False, xaxis_title=None, yaxis_title=var, xaxis={'categoryorder':'total ascending'})
                    fig.update_yaxes(showticklabels=True, showgrid=False,
                                     title='Densité', title_font=dict(size=18),
                                     tickfont=dict(size=18))
                    # fig.update_layout(width=380, height=300)
                    fig.update_layout(height=400)
                elif data_group[var].nunique() <= 2:
                    counts = data_group[var].value_counts()
                    counts_df = pd.DataFrame(
                        {'value': counts.index, 'count': counts.values})
                    fig = px.bar(counts_df, x='count', y='value',
                                 orientation='h',
                                 color_discrete_sequence=['forestgreen'])
                    client_val = data_client[var][0]
                    if var == 'CODE_GENDER':
                        val0 = 'F '
                        val1 = 'M '
                    else:
                        val0 = 'Non '
                        val1 = 'Oui '
                    fig.add_annotation(x=counts_df.loc[
                        counts_df['value'] == client_val, 'count'].iloc[0],
                                       y=client_val,
                                       ax=40, ay=0, text="Client", font_size=18,
                                       showarrow=True, arrowhead=1,
                                       arrowcolor='white')
                    fig.update_layout(
                        yaxis=dict(tickmode='array', ticktext=[val0, val1],
                                   tickvals=[0, 1]))
                    fig.update_layout(xaxis=dict(tickfont=dict(size=18)),
                                      yaxis=dict(tickfont=dict(size=18)))
                    fig.update_xaxes(title='Nombre de clients')
                    fig.update_yaxes(showticklabels=True, showgrid=False,
                                     title='')
                    fig.update_layout(height=400)
                # Show plot
                st.plotly_chart(fig, use_container_width=True,
                                theme=theme_plotly)

                # Show info about var and group 
                description = description_df.loc[
                    description_df['Row_name'] == var, 'Description'].values[0]
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


def main():
    API_endpoint = "http://13.37.31.96:8000/predict"

    # Get the cloud data location for the run of interest
    mlflow.set_tracking_uri("http://13.37.31.96:5000")
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name('MLflow_FinalModel')
    runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)
    run_id = \
    runs[runs['tags.mlflow.runName'] == 'LogisticRegression_'].run_id.values[0]
    run = client.get_run(run_id)
    artifacts_uri = run.info.artifact_uri
    metrics = run.data.metrics

    # Config
    st.set_page_config(page_title="Credit Score", page_icon=":sunglasses:",
                       layout="wide")

    # Title
    st.title('🏦 Prediction score crédit 💵')
    # st.markdown("<h1 style='text-align: center; color: black;'>Prediction score crédit</h1>", unsafe_allow_html=True)

    # Style
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # =========================== FETCH DATA & MODEL =========================== #
    Xtrain, Xtrain_clusters, Xtest_samples, Xtest_samples_clusters, description_df = fetch_data(
        artifacts_uri)
    model = fetch_model(artifacts_uri)
    explainer, shap_values = compute_shap(model, Xtrain)

    # Compute feature importances based on the SHAP values
    feature_importances_df = pd.DataFrame(np.mean(shap_values, axis=0),
                                          index=Xtrain.columns,
                                          columns=['SHAP'])
    feature_importances_df['abs_SHAP'] = feature_importances_df['SHAP'].abs()
    # Print feature importances
    # st.write(feature_importances_df)

    # =========================== DASHBOARD ITEMS =========================== #
    tab1, tab2, tab3, tab4 = st.tabs(['🗠 Visualiser les données du client',
                                      '💯 Calculer le score crédit du client',
                                      '📊 Explications du score',
                                      '⚙️ Informations sur le modèle'])

    # ======= Sidebar select client
    # Set the minimum width of the sidebar (default is 21rem)
    st.markdown(f'''
        <style>
            section[data-testid="stSidebar"] .css-ng1t4o {{width: 15rem;}}
            section[data-testid="stSidebar"] .css-1d391kg {{width: 15rem;}}
        </style>
    ''', unsafe_allow_html=True)
    clist = ['Client ' + str(x + 1) for x in range(len(Xtest_samples))]
    clist = ['<Sélectionnez un client>'] + clist
    client_id = clist[0]  # default
    client_id = st.sidebar.selectbox('Sélectionnez le client', clist, key=1000)
    if client_id == clist[0]:
        pass
    else:
        st.sidebar.subheader('Données de base')
        data_sample = Xtest_samples.iloc[[clist.index(client_id) - 1]]
        data_sample_cluster = Xtest_samples_clusters.iloc[
            [clist.index(client_id) - 1]]
        data_sample = data_sample.reset_index(drop=True)
        gender = data_sample['CODE_GENDER'].apply(
            lambda x: 'M' if x == 1.0 else 'F')
        st.sidebar.metric('Genre', '%s' % gender.values[0])
        st.sidebar.metric('Age', '%d ans' % np.ceil(
            -1 * data_sample['DAYS_BIRTH'] / 365.25))
        st.sidebar.metric('Revenus', '{:,}$'.format(
            int(data_sample['AMT_INCOME_TOTAL'].values[0])).replace(',', ' '))
        st.sidebar.metric("Nombre d'enfants",
                          '%d' % data_sample['CNT_CHILDREN'])

    # ======= Show client data
    with tab1:
        st.header('🗠 Visualiser les données du client')
        if client_id == clist[0]:
            st.write(
                '_Veuillez sélectionner un des clients dans le menu de la barre latérale_')
        else:
            st.subheader('Paramètres')
            # data_sample = Xtest_samples.iloc[[clist.index(client_id)-1]]
            # data_sample_cluster = Xtest_samples_clusters.iloc[[clist.index(client_id)-1]]
            # data_sample = data_sample.reset_index(drop=True)
            col1, col2, col3 = st.columns([1, 1.5, 0.5])
            with col1:
                options = ['Tous les clients',
                           'Même catégorie socio-démographique',
                           'Même groupe - proche sur toutes variables']
                helptxt = 'Comparer à tous les clients  \n' \
                          'Au groupe de clients similaires sur un sous-ensemble de variables socio-démographiques  \n' \
                          'Au groupe de clients similaires sur une combinaisons de toutes les varialbes'
                option = st.selectbox(
                    f'Sélectionnez la base de comparaison client', options,
                    help=helptxt)
                if option == options[0]:
                    Xgroup = Xtrain
                elif option == options[1]:
                    mask = Xtrain_clusters['cluster_SocioDemo'] == \
                           data_sample_cluster['cluster_SocioDemo'].values[0]
                    Xgroup = Xtrain[mask]
                elif option == options[2]:
                    mask = Xtrain_clusters['cluster_FullData'] == \
                           data_sample_cluster['cluster_FullData'].values[0]
                    Xgroup = Xtrain[mask]
            with col2:
                # helptxt = 'Toutes, ou limiter aux 6 plus importantes selon le modèle.  '\
                #           '\nPositives = font baisser le score.  \nNégatives = font augmenter le score'
                helptxt = 'Toutes, ou limiter aux 6 plus importantes selon le modèle'
                genre = st.radio("Quelles variables rechercher ?", ('Toutes',
                                                                    'Uniquement les plus importantes pour le score'),
                                 help=helptxt)
                if genre == 'Toutes':
                    varlist = sorted(data_sample.columns)
                elif genre == 'Uniquement les plus importantes pour le score':
                    feature_importances_df = feature_importances_df.sort_values(
                        by='abs_SHAP', ascending=False)
                    varlist = sorted(feature_importances_df[:15].index)
                # elif  genre == 'Uniquement les plus importantes pour le score (positives)':
                #     feature_importances_df = feature_importances_df.sort_values(by='SHAP', ascending=False)
                #     varlist = sorted(feature_importances_df[:6].index)
                # elif  genre == 'Uniquement les plus importantes pour le score (négatives)':
                #     feature_importances_df = feature_importances_df.sort_values(by='SHAP', ascending=True)
                #     varlist = sorted(feature_importances_df[:6].index)
                else:
                    varlist = sorted(data_sample.columns)[:10]  # top_features
            st.subheader('Graphiques')
            show_client_data(data_sample, Xgroup, description_df, varlist)

    # ======= Show client score prediction by API call
    with tab2:
        st.header('💯 Calculer le score crédit du client')
        if client_id == clist[0]:
            st.write(
                '_Veuillez sélectionner un des clients dans le menu de la barre latérale_')
        else:
            col1, col2 = st.columns([0.5, 1.5])
            # Predict button
            # thresold = st.number_input('Seuil de décision :', min_value=0, max_value=100, value=30, step=1, format='%d')
            # col1, col2, col3, col4, col5 = st.columns(5)  # trick to center elements --> use col3
            with col1:
                st.write('     \n')
                predict_btn = st.button('Calcul du score')
                threshold = metrics['cv_test_best_threshold']
                txt = 'Seuil de décision (probabilité de non-remboursement, recommandé = %d%%) :' % round(
                    threshold * 100)
                thresold = st.slider(txt, min_value=0, max_value=100,
                                     value=round(threshold * 100), step=1,
                                     format='%d%%')
            if predict_btn:
                print('Data is :')
                print(data_sample)
                # pred = model.predict_proba(data_sample)[0][1]  # using downloaded model
                # using API on the server
                response = requests.post(API_endpoint,
                                         json=data_sample.to_dict())
                pred = response.json()[0]
                with col1:
                    st.metric('Score :', value=('%d/100' % ((1 - pred) * 100)))
                    st.write('Probabilité de ne pas rembourser: %.01f%%' % (
                                pred * 100))
                    st.sidebar.metric('Score :',
                                      value=('%d/100' % ((1 - pred) * 100)))
                with col2:
                    fig = color_gauge(pred, threshold)
                    st.pyplot(fig)
                    if pred >= (thresold / 100):
                        st.markdown(
                            '<h1 style="text-align:center;color:red;font-weight:700;font-size:40px">Crédit refusé !!! 👎😦</h1>',
                            unsafe_allow_html=True)
                    elif pred < (thresold / 100):
                        st.markdown(
                            '<h1 style="text-align:center;color:green;font-weight:700;font-size:40px">Crédit accordé !!! 🏆🤑</h1>',
                            unsafe_allow_html=True)
                        st.balloons()

    # ======= Score explanations
    with tab3:
        st.header('📊 Explications du score')
        if client_id == clist[0]:
            st.write(
                '_Veuillez sélectionner un des clients dans le menu de la barre latérale_')
        else:
            col1, col2, col3, col4 = st.columns(4)
            # with col1:
            #     shap_btn = st.button('📊 Générer les explications du score')
            # if shap_btn:

            # Compute SHAP values for the client
            shap_values_client = explainer.shap_values(data_sample)

            # Shap values for the sample client in a df
            shap_sample_df = pd.DataFrame(shap_values_client[0],
                                          index=data_sample.columns,
                                          columns=['SHAP'])
            shap_sample_df['asb_SHAP'] = shap_sample_df['SHAP'].abs()
            shap_sample_df = shap_sample_df.sort_values(by='asb_SHAP',
                                                        ascending=False)
            # Add descriptions
            shap_sample_df = shap_sample_df.merge(description_df,
                                                  left_index=True,
                                                  right_on='Row_name',
                                                  how='left')

            # Display features descriptions:
            col1, col2 = st.columns(2)
            with col1:
                st.write(
                    'Les variables suivantes font augmenter le score de ce client 👍📈 (diminuer la probabilité de non-remboursement) :')
                shap_negat_df = shap_sample_df.sort_values(by='SHAP',
                                                           ascending=True)[:10]
                st.write(shap_negat_df[['Row_name', 'SHAP', 'Description']])
                # st.markdown(shap_negat_df[['Row_name', 'SHAP', 'Description']].style.hide(axis="index").to_html(), unsafe_allow_html=True)
            with col2:
                st.write(
                    'Les variables suivantes font diminuer le score de ce client 👎📉 (augmenter la probabilité de non-remboursement) :')
                shap_posit_df = shap_sample_df.sort_values(by='SHAP',
                                                           ascending=False)[:10]
                st.write(shap_posit_df[['Row_name', 'SHAP',
                                        'Description']])  # nicer but shows index...
                # st.markdown(shap_posit_df[['Row_name', 'SHAP', 'Description']].style.hide(axis="index").to_html(), unsafe_allow_html=True)  # in markdown without index

            # Force plot
            st.subheader('Force plot')
            # fig, ax = plt.subplots(facecolor='None')
            fig = shap.force_plot(explainer.expected_value, shap_values_client,
                                  data_sample)
            st_shap(fig, width=1200)  # Display the plot in the Streamlit app

            # Waterfall_plot
            st.subheader('Waterfall plot')
            fig, ax = plt.subplots(facecolor='None')
            ax.set_facecolor("None")
            shap.waterfall_plot(shap.Explanation(values=shap_values_client[0],
                                                 base_values=explainer.expected_value,
                                                 data=data_sample.iloc[0]),
                                max_display=15)
            plt.setp(ax.get_xticklabels(), color="white")
            plt.setp(ax.get_yticklabels(), color="white")
            # for spine in ax.spines.values():
            #     spine.set_color('white')
            st_shap(fig, height=500, width=1000)

    # ======= Show model info
    with tab4:
        st.header('⚙️ Informations sur le modèle')
        st.write('Nom du run : ' + run.info.run_name)

        # Metrics
        st.subheader('Métriques')
        cols = st.columns(10)
        with cols[0]:
            st.metric('Score métier',
                      np.round(metrics['cv_test_custom_score'], 3))
        with cols[1]:
            st.metric('ROC AUC', np.round(metrics['cv_test_roc_auc'], 3))
        with cols[2]:
            st.metric('F1 score', np.round(metrics['cv_test_f1'], 3))
        with cols[3]:
            st.metric('Accuracy', np.round(metrics['cv_test_accuracy'], 3))

        # Summary plots
        col1, col2 = st.columns(2)
        matplotlib.rcParams['text.color'] = 'white'
        with col1:
            st.subheader('Importance des features')
            fig, ax = plt.subplots(facecolor='None')
            shap.summary_plot(shap_values, features=Xtrain[:50],
                              plot_type="bar", max_display=15,
                              color='forestgreen')
            # Set colors to white
            ax.set_facecolor("None")
            plt.setp(ax.get_xticklabels(), color="white", fontsize=10)
            plt.setp(ax.get_yticklabels(), color="white", fontsize=10)
            for spine in ax.spines.values():
                spine.set_color('white')
            st_shap(fig)
        with col2:
            st.subheader('Distribution de la contribution des features')
            fig, ax = plt.subplots(facecolor='None')
            shap.summary_plot(shap_values, features=Xtrain[:50],
                              plot_type='violin', max_display=15)
            # Set colors to white
            ax.set_facecolor("None")
            plt.setp(ax.get_xticklabels(), color="white", fontsize=10)
            plt.setp(ax.get_yticklabels(), color="white", fontsize=10)
            for spine in ax.spines.values():
                spine.set_color('white')
            st_shap(fig)

    st.markdown("---")


if __name__ == '__main__':
    main()
