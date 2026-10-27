"""Entraîne Football Match Predictor avec 10 variables des 5 matchs précédents.

Lancement : python train_model_corrige.py
Mettre ce script dans `ml_streamlit_docker/` (à côté de app.py),
et les données dans `../data/football_2025_2027.csv`.
"""
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_NAME = "football_2025_2027.csv"
FEATURE_COLUMNS = [
    "Home_Form", "Home_Wins", "Home_Goals_Avg",
    "Home_Conceded_Avg", "Home_Matches_Played",
    "Away_Form", "Away_Wins", "Away_Goals_Avg",
    "Away_Conceded_Avg", "Away_Matches_Played",
]

# 1. Charger les données (fonctionne depuis la racine ou ml_streamlit_docker/)
data_candidates = [
    BASE_DIR.parent / "data" / DATA_NAME,
    BASE_DIR / "data" / DATA_NAME,
    BASE_DIR / DATA_NAME,
]
data_path = next((p for p in data_candidates if p.is_file()), None)
if data_path is None:
    raise FileNotFoundError(
        f"Impossible de trouver {DATA_NAME} dans data/. "
        "Vérifier l'emplacement du CSV."
    )

raw = pd.read_csv(data_path, low_memory=False)
required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
missing = sorted(set(required) - set(raw.columns))
if missing:
    raise ValueError(f"Colonnes absentes : {missing}")

df = raw[required].copy()
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
for col in ["FTHG", "FTAG"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["FTR"] = df["FTR"].astype("string").str.strip().str.upper()
df = df[df["FTR"].isin(["H", "D", "A"])].copy()
df = df.dropna(subset=required)
df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
df = df.sort_values("Date").reset_index(drop=True)

if len(df) < 100:
    raise ValueError("Pas assez de matchs pour entraîner ce modèle.")
print(f"Dataset : {len(df)} matchs, du {df['Date'].min():%d/%m/%Y} au {df['Date'].max():%d/%m/%Y}")

# 2. Calculer les 10 features AVANT chaque match
history = defaultdict(list)

def stats(team):
    recent = history[team][-5:]
    n = len(recent)
    return [
        sum(m[0] for m in recent),                  # Points sur les 5 derniers matchs
        sum(m[1] for m in recent),                  # Victoires
        sum(m[2] for m in recent) / n if n else 0, # Buts marqués / match
        sum(m[3] for m in recent) / n if n else 0, # Buts encaissés / match
        n,                                          # Nombre de matchs connus
    ]

rows = []
for match in df.itertuples(index=False):
    home, away = match.HomeTeam, match.AwayTeam
    rows.append(stats(home) + stats(away))

    home_pts = 3 if match.FTR == "H" else (1 if match.FTR == "D" else 0)
    away_pts = 3 if match.FTR == "A" else (1 if match.FTR == "D" else 0)
    history[home].append((home_pts, int(match.FTR == "H"), match.FTHG, match.FTAG))
    history[away].append((away_pts, int(match.FTR == "A"), match.FTAG, match.FTHG))

X = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
y = df["FTR"].astype(str)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("Features :", X.shape, "Classes :", list(encoder.classes_))

# 3. Split chronologique (20 % des derniers matchs uniquement pour tester)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y_encoded[:split_idx], y_encoded[split_idx:]

if len(np.unique(y_train)) != 3:
    raise ValueError("Le jeu d'entraînement n'a pas les 3 résultats H, D et A.")

# 4. Réglage des hyperparamètres : validation temporelle, PAS de mélange
search = RandomizedSearchCV(
    estimator=XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss",
        random_state=42, n_jobs=2,
    ),
    param_distributions={
        "n_estimators": [60, 100, 150],
        "max_depth": [2, 3],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1, 3, 5],
    },
    n_iter=8,
    cv=TimeSeriesSplit(n_splits=3),
    scoring="neg_log_loss",
    n_jobs=1,
    random_state=42,
    error_score="raise",
)
search.fit(X_train, y_train)
model = search.best_estimator_

# 5. Evaluation honnête sur les derniers matchs du test
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("\nMeilleurs paramètres :", search.best_params_)
print("Accuracy Baseline :", round(accuracy_score(y_test, baseline.predict(X_test)), 4))
print("Accuracy XGBoost  :", round(accuracy_score(y_test, preds), 4))
print("Log Loss         :", round(log_loss(y_test, probs, labels=model.classes_), 4))
print("\nRapport de classification :")
print(classification_report(y_test, preds, labels=range(3), target_names=encoder.classes_, zero_division=0))

# 6. Réentraînement sur toutes les données pour l'application
final_model = XGBClassifier(
    objective="multi:softprob", eval_metric="mlogloss", random_state=42,
    n_jobs=2, **search.best_params_,
)
final_model.fit(X, y_encoded)

# Emplacement identique à app.py : ml_streamlit_docker/models/
if (BASE_DIR / "app.py").exists():
    model_dir = BASE_DIR / "models"
elif (BASE_DIR / "ml_streamlit_docker" / "app.py").exists():
    model_dir = BASE_DIR / "ml_streamlit_docker" / "models"
else:
    model_dir = BASE_DIR / "models"
model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(final_model, model_dir / "football_model.pkl")
joblib.dump(encoder, model_dir / "label_encoder.pkl")
joblib.dump(FEATURE_COLUMNS, model_dir / "feature_columns.pkl")
print("\nModèle et encodeur enregistrés dans :", model_dir)
print("Remarque : les scores du test correspondent au modèle évalué AVANT le réentraînement final.")
