import pandas as pd
import streamlit as st
import requests
import numpy as np
import shap
import mlflow
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
import plotly.express as px


def show_client_data(data_client, data_group, description_df, varlist):
    cols = st.columns(4)
    for i in range(len(cols)):
        with cols[i]:
            # if i == 0:
            #     col1.subheader('Histogrammes des variables')
            #     st.write('')
            options = ['<Sélectionnez une variable>'] + varlist
            value = options[0]  # default
            var = st.selectbox(f'Sélectionnez la variable à visualiser', options, key=i)
            if var != options[0]:
                fig = px.histogram(data_group, x=var, log_x=False, histnorm="density",
                                   color_discrete_sequence=['forestgreen']) # nbins=60, 
                fig.add_vline(x=data_client[var][0], line_dash = 'solid', line_color = 'red', line_width=3,
                                 annotation=dict(
                                        text="Client",
                                        font_size=16,
                                        showarrow=True,
                                        arrowhead=1,
                                        ax=0,
                                        ay=-20), annotation_position='top')
                fig.update_yaxes(showticklabels=False, showgrid=False, title='')
                fig.update_layout(width=380, height=300)
                      
                # Show plot
                st.plotly_chart(fig)

                # Show info about var and group 
                description = description_df.loc[description_df['Row_name'] == var, 'Description'].values[0]
                client_val = data_client[var][0]
                group_mean = data_group[var].mean()
                group_std = data_group[var].std()
                group_median = data_group[var].median()
                rows = [
                    f'Description: {description}\n',
                    f'Client: {client_val:.02f}\n',
                    f'Moyenne groupe: {group_mean:.02f} (+/- {group_std:.02f})\n',
                    f'Médiane groupe: {group_median:.02f}\n'
                ]
                st.write('\n'.join(rows))
            

            
def main():
    MLFLOW_URI = 'http://127.0.0.1:1234/invocations'
    run_id = '8d9d222ba56b453c9bf1943bff24aa70'
 
    st.set_page_config(page_title="Credit Score", page_icon=":sunglasses:", layout="wide")

    # =========================== DATA & COMPUTATIONS =========================== #
    # Load sample clients
    art_uri = 'runs:/'+run_id+'/Xtest_samples.csv'
    Xtest_samples = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri))
    art_uri = 'runs:/'+run_id+'/Xtest_samples.csv'
    Xtest_samples = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri))
    art_uri = 'runs:/'+run_id+'/Xtest_samples_clusters.csv'
    Xtest_samples_clusters = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri))

    # Load features description
    desc_uri = 'runs:/'+run_id+'/description_df.csv'
    description_df = pd.read_csv(mlflow.artifacts.download_artifacts(desc_uri))
    
    # Load the saved model from MLflow
    model_uri = art_uri = 'runs:/'+run_id+'/model'
    st.write('Using model from: ' + model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    # Load the training data
    art_uri = 'runs:/'+run_id+'/Xtrain.csv'
    Xtrain = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri))
    art_uri = 'runs:/'+run_id+'/Xtrain_clusters.csv'
    Xtrain_clusters = pd.read_csv(mlflow.artifacts.download_artifacts(art_uri))
    
    # Load some metrics
    run = mlflow.get_run(run_id)
    Optimal_threshold = run.data.metrics['Optimal_threshold']
    
    # # Compute SHAP values using training data & model
    # # model_step = model.named_steps['Model']
    # model_step = model
    # def predict_fn(x): return model_step.predict_proba(x)[:, 1]  # use predict_proba for binary classification
    # explainer = shap.KernelExplainer(predict_fn, shap.kmeans(Xtrain, 20))
    
    # =========================== DASHBOARD ITEMS =========================== #   
    # st.title('Prediction score crédit')
    st.markdown("<h1 style='text-align: center; color: black;'>Prediction score crédit</h1>", unsafe_allow_html=True)
    st.markdown("---") 
    
    tab1, tab2 = st.tabs(['Visualiser les données du client', 'Calculer le score crédit du client'])
    
    # Side bar select client
    clist = ['Client '+str(x+1) for x in range(len(Xtest_samples))]
    clist = ['<Sélectionnez un client>'] + clist
    client_id = clist[0]  # default
    client_id = st.sidebar.selectbox('Sélectionnez le client', clist, key=1000)

    # ======= Show client data
    with tab1:
        st.header('Visualiser les données du client')
        # # Inputs
        # clist = ['Client '+str(x+1) for x in range(len(Xtest_samples))]
        # clist = ['<Sélectionnez un client>'] + clist
        # value = clist[0]  # default
        col1, col2, col3, col4 = st.columns(4) 
        with col1:
            # value = st.selectbox(f'Sélectionnez le client', clist, key=1000)
            if client_id != '<Sélectionnez un client>' :
                data_sample = Xtest_samples.iloc[[clist.index(client_id)]]
                data_sample_cluster = Xtest_samples_clusters.iloc[[clist.index(client_id)]]
                data_sample = data_sample.reset_index(drop=True)
                # st.write(data_sample)
        if client_id != clist[0]:
            with col1:
                options = ['Tous les clients', 'Même catégorie socio-démographique', 'Même groupe - toutes variables']
                option = st.selectbox(f'Sélectionnez la base de comparaison client', options)
                if option == options[0]:
                    Xgroup = Xtrain
                elif option == options[1]:
                    mask = Xtrain_clusters['cluster_SocioDemo'] == data_sample_cluster['cluster_SocioDemo'].values[0]
                    Xgroup = Xtrain[mask]
                elif option == options[2]:
                    mask = Xtrain_clusters['cluster_FullData'] == data_sample_cluster['cluster_FullData'].values[0]
                    Xgroup = Xtrain[mask]
            with col2:
                genre = st.radio("Quelles variables rechercher ?",('Toutes', 'Uniquement les plus importantes pour le score'))
                if genre == 'Toutes':
                    varlist = sorted(data_sample.columns)
                else:
                    varlist = sorted(data_sample.columns)[:10] #top_features
            show_client_data(data_sample, Xgroup, description_df, varlist)
        else:
            st.write('Veuillez sélectionner un des clients dans le menu de la barre latérale')
    st.markdown("---") 
    
    # ======= Show client score prediction
    with tab2:
        st.header('Calculer le score crédit du client')
        col1, col2, col3, col4 = st.columns(4) 
        with col1:
            # value = st.selectbox(f'Sélectionnez le client', clist, key=1001)
            if client_id != '<Sélectionnez un client>' :
                data_sample = Xtest_samples.iloc[[clist.index(client_id)]]
                data_sample_cluster = Xtest_samples_clusters.iloc[[clist.index(client_id)]]
                data_sample = data_sample.reset_index(drop=True)
                # st.write(data_sample)
        if client_id != clist[0]:
            # Predict button
            # thresold = st.number_input('Seuil de décision :', min_value=0, max_value=100, value=30, step=1, format='%d') 
            # col1, col2, col3, col4, col5 = st.columns(5)  # trick to center elements --> use col3
            with col1:
                txt = 'Seuil de décision (probabilité de non-remboursement, recommandé = %d%%) :' % round(Optimal_threshold*100)
                thresold = st.slider(txt, min_value=0, max_value=100, value=round(Optimal_threshold*100), step=1, format='%d%%') 
            with col3:
                predict_btn = st.button('Calcul du score')
            if predict_btn:
                print('Data is :')
                print(data_sample)
                pred = model.predict_proba(data_sample)[0][1]
                with col3:
                    st.metric('Score :', value=('%d/100' % ((1-pred)*100)))
                    st.write('Probabilité de ne pas rembourser: %.01f%%' % (pred*100))
                if pred>=(thresold/100):
                    st.markdown('<h1 style="text-align:center;color:red;font-weight:700;font-size:40px">Crédit refusé !!! 👎😦</h1>', unsafe_allow_html=True)
                elif pred<(thresold/100):
                    st.markdown('<h1 style="text-align:center;color:green;font-weight:700;font-size:40px">Crédit accordé !!! 🏆🤑</h1>', unsafe_allow_html=True)
                    st.balloons()

            st.markdown("---")    
            st.subheader('Explications du score:')
            
            
            
            
        else:
            st.write('Veuillez sélectionner un des clients dans le menu de la barre latérale')
        # st.markdown("<h2 style='text-align: center; color: grey;'>Explications :</h2>", unsafe_allow_html=True)
        
#         # Compute SHAP values for the client
#         shap_values_client = explainer.shap_values(data_sample)
 
#         # Force plot
#         fig, ax = plt.subplots(facecolor='None')
#         fig = shap.force_plot(explainer.expected_value, shap_values_client, data_sample)
#         st_shap(fig) # Display the plot in the Streamlit app
        
#         # Waterfall_plot
#         shap_values_client = explainer.shap_values(data_sample)
#         fig, ax = plt.subplots(facecolor='None')
#         shap.waterfall_plot(shap.Explanation(values=shap_values_client[0],
#                                              base_values=explainer.expected_value,
#                                              data=data_sample.iloc[0]), max_display=15)
#         st.pyplot(fig) # Display the plot in the Streamlit app
        

#         # Shap values for the sample client in a df
#         shap_sample_df = pd.DataFrame(shap_values_client[0], index=data_sample.columns, columns=['SHAP'])
#         shap_sample_df['asb_SHAP'] = shap_sample_df['SHAP'].abs()
#         shap_sample_df = shap_sample_df.sort_values(by='asb_SHAP', ascending=False)
#         # Add descriptions
#         shap_sample_df = shap_sample_df.merge(description_df, left_index=True, right_on='Row_name', how='left')
        
#         # Display features description:
#         st.write(shap_sample_df.iloc[:15][['Row_name', 'SHAP', 'Description']])
#         col1, col2, col3, col4, col5 = st.columns(5)
#         k = 0
#         for col in [col1, col2, col3, col4, col5]:
#             with col:
#                 for i in range(k, k+3, 1):
#                     st.metric(shap_sample_df.iloc[i]['Row_name'],
#                               np.round(shap_sample_df.iloc[i]['SHAP'], 3),
#                               help=shap_sample_df.iloc[i]['Description'])
#                     k+=2
    st.markdown("---") 
if __name__ == '__main__':
    main()
