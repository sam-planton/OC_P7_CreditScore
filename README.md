![workflow status](https://github.com/sam-planton/OC_P7_CreditScore/actions/workflows/deploy_API&dashboard_workflow.yml/badge.svg) 
[![CC BY 4.0][cc-by-shield]][cc-by]
# OC P7 CreditScore 🏦
Cette application, réalisée dans le cadre d'un projet de formation, vise à attribuer un "score crédit" à un client demandant un crédit bancaire (et dont un certain nombre d'informations sont connues). Ce score est basé sur la probabilité de défaut du client, déterminée à partir d'un modèle de machine learning.

Les données utilisées proviennent du jeu de données Kaggle
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview)

Lien du dashboard en ligne : 👉 https://sp-oc-p7-dashboard.herokuapp.com/ 👈

## Modèle
Le modèle utilisé est un modèle de classification utilisant le gradient boosting : "LGBMClassifier". Il est entraîné et optimisé en utilisant les librairies [scikit-learn](https://scikit-learn.org/) et [LightGBM](https://lightgbm.readthedocs.io/).

## API de prédiction
Une API de prédiction est déployée via [Heroku](https://www.heroku.com/). Il s'agit d'une application Flask utilisant le modèle entraîné et renvoyant, à partir des données d'un client fournies, la probabilité de défaut prédite.

## Dashboard
Un dashboard réalisé sous [Streamlit](https://streamlit.io/) et déployé via [Heroku](https://www.heroku.com/) permet d'explorer les données et de récupérer les prédictions du modèles à partir de l'API pour un échantillon de clients fourni. Des explications sur les prédictions locales et globales (importance des features, via librairie SHAP) sont aussi fournies.

## Principaux fichiers
- `api.py` : application Flask de l'API de prédiction. Le fichier `\api_proc\Procfile` sert au déploiement.
- `dashboard.py` : application Streamlit. Le fichier `\dashboard_proc\Procfile` sert au déploiement.
- Dossier `\data\` contient les données d'entraînement prétraitées ainsi que d'autres de données utilisées dans le dasboard (ex: données clients test)
- Dossier `\data\model` contient le modèle entraîné, exporté via [MLflow](https://mlflow.org/)

---
[![CC BY 4.0][cc-by-image]][cc-by] This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].



[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
