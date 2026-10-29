# ⚽ Football Match Predictor — Premier League

Application de **prédiction des résultats de matchs de Premier League** réalisée avec Python et le Machine Learning. Elle estime les probabilités de trois issues : **victoire à domicile (H)**, **match nul (D)** et **victoire à l'extérieur (A)**.

Le projet combine un travail de préparation des données, un modèle de classification, une interface web **Streamlit**, une API **FastAPI** et un environnement **Docker**.

> **Important :** les prédictions sont expérimentales. Les probabilités produites par le modèle ne garantissent pas le résultat réel d'un match.

## 🎯 Objectifs

- Exploiter les données historiques de la Premier League au format CSV.
- Nettoyer et préparer les données avant l'entraînement.
- Construire des variables décrivant la forme récente des équipes.
- Entraîner un modèle capable de distinguer les résultats **H / D / A**.
- Évaluer ses performances et le comparer à des modèles de référence.
- Permettre une prédiction depuis une interface web et une API REST.
- Conteneuriser l'application avec Docker.

## 🛠️ Technologies utilisées

| Domaine | Technologies |
| --- | --- |
| Langage | Python 3.11 |
| Données | Pandas, NumPy |
| Machine Learning | scikit-learn, XGBoost |
| Enregistrement du modèle | Joblib |
| Interface web | Streamlit |
| API REST | FastAPI, Uvicorn |
| Déploiement local | Docker, Docker Compose |
| Développement | VS Code, Jupyter Notebook |

## 📊 Données utilisées

- **Source :** [Football-Data.co.uk — England](https://www.football-data.co.uk/englandm.php).
- **Fichier principal :** `football_2025_2027.csv`.
- **Version étudiée :** 450 matchs et 177 colonnes, provenant des saisons 2025/2026 et du début de 2026/2027.
- **Cible (`FTR`) :** `H` (domicile), `D` (nul), `A` (extérieur).

Le fichier CSV est une **copie locale** : les matchs ne se mettent pas automatiquement à jour depuis le site source.

### Nettoyage des données

1. Sélection des colonnes utiles : date, équipes, score final, résultat et quelques statistiques de match.
2. Conversion des dates en `datetime` et des valeurs numériques en nombres.
3. Normalisation des valeurs `FTR` (`H`, `D`, `A`).
4. Suppression des lignes dont les informations **essentielles** sont manquantes.
5. Suppression des matchs dupliqués et tri chronologique.

Les scores et statistiques du **match à prédire** ne sont pas fournis directement au modèle : ils servent uniquement à construire l'historique des matchs déjà joués, afin d'éviter une fuite de données (*data leakage*).

## 🧠 Feature Engineering

Pour chaque rencontre, le programme calcule les statistiques disponibles sur les **cinq matchs précédents** de chacune des deux équipes. Les variables sont calculées **avant** d'ajouter le résultat de la rencontre courante à l'historique.

| Équipe à domicile | Équipe à l'extérieur | Signification |
| --- | --- | --- |
| `Home_Form` | `Away_Form` | Total des points sur les cinq derniers matchs |
| `Home_Wins` | `Away_Wins` | Nombre de victoires |
| `Home_Goals_Avg` | `Away_Goals_Avg` | Moyenne de buts marqués |
| `Home_Conceded_Avg` | `Away_Conceded_Avg` | Moyenne de buts encaissés |
| `Home_Matches_Played` | `Away_Matches_Played` | Nombre de matchs précédents disponibles (0 à 5) |

**Total : 10 features par match.** Les cinq dernières rencontres incluent les matchs à domicile **et** à l'extérieur.

## 🤖 Entraînement et évaluation

Le notebook `football_model.ipynb` permet de suivre les principales étapes :

1. Préparation de `X` (10 features) et `y` (`FTR`).
2. **Découpage chronologique 80 % entraînement / 20 % test**.
3. Encodage des classes à l'aide de `LabelEncoder` : `A = 0`, `D = 1`, `H = 2`.
4. Entraînement de **XGBoost** et comparaison avec **Logistic Regression** et un **DummyClassifier** (baseline majoritaire).
5. Évaluation par **accuracy**, **classification report** (precision, recall, F1-score) et **confusion matrix**.
6. Enregistrement du modèle et des métadonnées nécessaires à la prédiction.

Le script `train_model.py` peut être utilisé pour automatiser l'entraînement. La version corrigée utilise **XGBoost**, une recherche d'hyperparamètres avec `TimeSeriesSplit` et une métrique de validation adaptée (`neg_log_loss`). Après l'évaluation, le modèle final est réentraîné sur l'ensemble des matchs disponibles pour l'application.

**Limite actuelle :** les premiers tests ont montré une précision encore insuffisante, notamment pour distinguer les matchs nuls. Le projet constitue un prototype pédagogique, et non un système de pronostics fiable.

## 📁 Structure du projet

Organisation attendue avec l'interface et l'API :

```text
mini projet ml/
├── data/
│   └── football_2025_2027.csv
├── football_model.ipynb
└── ml_streamlit_docker/
    ├── app.py                  # Interface Streamlit
    ├── api.py                  # API FastAPI
    ├── train_model.py          # Entraînement du modèle
    ├── requirements.txt
    ├── Dockerfile
    ├── docker-compose.yml
    └── models/
        ├── football_model.pkl
        ├── label_encoder.pkl
        └── feature_columns.pkl
```

> Les noms et chemins peuvent varier selon l'organisation locale. Le fichier `app.py` doit se trouver dans le dossier monté sous `/app` par Docker ; le CSV est monté sous `/data` dans la configuration Docker Compose présentée ici.

## 🚀 Installation et exécution

### 1. Prérequis

- Python 3.11 (pour l'exécution locale).
- Docker Desktop (pour l'exécution via conteneurs).
- Les fichiers du modèle dans `models/` et le CSV dans `../data/`.

### 2. Installer les dépendances

Depuis le dossier `ml_streamlit_docker` :

```powershell
python -m pip install -r requirements.txt
```

Vérifier que `requirements.txt` contient notamment `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, **`fastapi`** et **`uvicorn`** pour lancer les deux services.

### 3. Entraîner le modèle si nécessaire

```powershell
python train_model.py
```

Cette étape génère dans `models/` :

- `football_model.pkl` — modèle entraîné ;
- `label_encoder.pkl` — correspondance entre classes et nombres ;
- `feature_columns.pkl` — noms et ordre des 10 features.

Si le script conservé porte le nom `train_model_corrige.py`, utiliser ce nom dans la commande.

### 4. Exécution locale sans Docker

Dans un premier terminal :

```powershell
python -m streamlit run app.py
```

Dans un second terminal :

```powershell
python -m uvicorn api:app --reload --port 8000
```

- Interface Streamlit : <http://localhost:8501>
- Documentation interactive de l'API : <http://localhost:8000/docs>

### 5. Exécution avec Docker Compose

La version Docker Compose à deux services contient **`web`** (Streamlit) et **`api`** (FastAPI), avec les ports `8501` et `8000` :

```powershell
docker compose up --build -d
```

Vérifier les services :

```powershell
docker compose ps
docker compose logs -f
```

Arrêter les conteneurs :

```powershell
docker compose down
```

> **Remarque :** l'ancienne configuration Docker Compose comprenait un service MySQL. Il n'est pas nécessaire à cette version de l'application, qui travaille avec des fichiers CSV et des modèles sauvegardés. Pour avoir les deux ports, utiliser la configuration Compose comprenant `web` et `api`.

## 🌐 API REST

La version FastAPI expose les endpoints suivants :

| Méthode | Route | Description |
| --- | --- | --- |
| `GET` | `/health` | Vérifier que l'API répond |
| `GET` | `/teams` | Obtenir les équipes disponibles dans le CSV |
| `POST` | `/predict` | Estimer les probabilités d'un match |
| `GET` | `/docs` | Tester l'API via Swagger UI |

### Exemple de requête

```http
POST http://localhost:8000/predict
Content-Type: application/json
```

```json
{
  "home_team": "Arsenal",
  "away_team": "Liverpool"
}
```

La réponse inclut : `home_team`, `away_team`, `home_win`, `draw`, `away_win`, `prediction` et `note`.

Les valeurs `home_win`, `draw` et `away_win` sont des probabilités comprises entre **0 et 1** ; `prediction` est la classe la plus probable (`H`, `D` ou `A`). Les chiffres dépendent du modèle enregistré et des matchs disponibles dans le CSV.

## 🖥️ Interface Streamlit

L'interface permet de :

- sélectionner une équipe à domicile et une équipe à l'extérieur ;
- calculer leurs indicateurs récents à partir du CSV ;
- cliquer sur **Générer la Prédiction** ;
- afficher les probabilités de victoire à domicile, de match nul et de victoire à l'extérieur.

## ⚠️ Limites et améliorations possibles

- Peu de données pour un problème de prédiction sportive complexe.
- Performances à comparer systématiquement à une baseline, et à évaluer sur des périodes futures.
- Les résultats de match nul restent difficiles à prédire.
- Les probabilités ne sont pas nécessairement calibrées.
- Pas de mise à jour automatique du CSV dans la version actuelle.
- Améliorations envisageables : davantage de saisons, statistiques domicile/extérieur séparées, classement et force des équipes, validation temporelle plus robuste, actualisation des données.

## 📚 Source

[Football-Data.co.uk — English football datasets](https://www.football-data.co.uk/englandm.php)

---

*Projet pédagogique de Machine Learning — Python, XGBoost, Streamlit, FastAPI et Docker.*
