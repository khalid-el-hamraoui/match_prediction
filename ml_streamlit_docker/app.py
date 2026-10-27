"""Streamlit UI for 10-feature Football Match Predictor (last five completed matches)."""
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILENAME = "football_2025_2027.csv"  # Dataset utilisé pour entraîner le modèle
FEATURES = [
    "Home_Form", "Home_Wins", "Home_Goals_Avg", "Home_Conceded_Avg",
    "Home_Matches_Played", "Away_Form", "Away_Wins", "Away_Goals_Avg",
    "Away_Conceded_Avg", "Away_Matches_Played",
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp {background: #0f172a; color: #f8fafc;}
html, body {font-family: 'Inter', sans-serif;}
.hero-banner {
  background:linear-gradient(135deg,#1e1b4b,#312e81 50%,#4338ca);
  border:1px solid rgba(255,255,255,.12);border-radius:18px;
  padding:30px 22px;text-align:center;margin-bottom:22px;
}
.hero-title {font-size:34px;font-weight:700;color:white;}
.hero-subtitle {color:#c7d2fe;margin-top:4px;}
.prob-dashboard {display:flex;gap:14px;margin:20px 0;flex-wrap:wrap;}
.prob-box {flex:1 1 175px;border-radius:14px;padding:22px 12px;text-align:center;color:white;}
.box-home {background:linear-gradient(135deg,#065f46,#10b981);}
.box-draw {background:linear-gradient(135deg,#334155,#64748b);}
.box-away {background:linear-gradient(135deg,#9f1239,#f43f5e);}
.prob-title {font-size:14px;font-weight:600;}
.prob-score {font-size:38px;font-weight:700;margin-top:7px;}
.stButton > button[kind="primary"] {background:#4f46e5;color:white;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
  <div class="hero-title">⚽ Premier League Match Predictor</div>
  <div class="hero-subtitle">Prédiction à partir des 5 derniers matchs de chaque équipe</div>
</div>
""", unsafe_allow_html=True)


def find_data_file():
    """Supporte app.py dans ml_streamlit_docker/ ou à la racine du projet."""
    candidates = [
        Path("/data") / DATA_FILENAME,
        BASE_DIR.parent / "data" / DATA_FILENAME,
        BASE_DIR / "data" / DATA_FILENAME,
        BASE_DIR / DATA_FILENAME,
    ]
    return next((p for p in candidates if p.exists()), None)


@st.cache_resource
def load_model():
    model_dir = BASE_DIR / "models"
    paths = [model_dir / "football_model.pkl", model_dir / "label_encoder.pkl"]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            "Fichiers manquants dans models/: football_model.pkl et label_encoder.pkl. "
            "Exécute la Cell 10 du notebook."
        )
    model, encoder = (joblib.load(p) for p in paths)
    column_file = model_dir / "feature_columns.pkl"
    column_names = joblib.load(column_file) if column_file.exists() else FEATURES
    if list(column_names) != FEATURES:
        raise ValueError("Les colonnes du modèle ne correspondent pas aux 10 Features attendues.")
    if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
        raise ValueError("Ce modèle attend un autre nombre de Features. Réexécute la Cell 10.")
    return model, encoder


@st.cache_data
def read_matches(path_str, file_time):
    data = pd.read_csv(path_str, low_memory=False)
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Colonnes absentes dans le CSV: {missing}")
    data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y", errors="coerce")
    for col in ("FTHG", "FTAG"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["FTR"] = data["FTR"].astype("string").str.strip().str.upper()
    data = data[data["FTR"].isin(["H", "D", "A"])].copy()
    data = data.dropna(subset=required)
    data = data.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    return data.sort_values("Date").reset_index(drop=True)


def recent_form(data, team):
    """Même logique que le Notebook: 5 derniers matchs, à domicile ou à l'extérieur."""
    last = data[(data["HomeTeam"] == team) | (data["AwayTeam"] == team)]
    last = last.tail(5).iloc[::-1]  # Le dataset est déjà trié du plus ancien au récent.
    points, wins, goals, conceded = 0, 0, 0.0, 0.0
    details = []
    for _, game in last.iterrows():
        at_home = game["HomeTeam"] == team
        result = str(game["FTR"])
        victory = result == ("H" if at_home else "A")
        draw = result == "D"
        gp = int(game["FTHG"] if at_home else game["FTAG"])
        gc = int(game["FTAG"] if at_home else game["FTHG"])
        p = 3 if victory else (1 if draw else 0)
        points += p
        wins += int(victory)
        goals += gp
        conceded += gc
        details.append({
            "Date": game["Date"].strftime("%d/%m/%Y"),
            "Match": f"{game['HomeTeam']} – {game['AwayTeam']}",
            "Score": f"{int(game['FTHG'])}-{int(game['FTAG'])}",
            "Résultat": "V" if victory else ("N" if draw else "D"),
        })
    n = len(last)
    values = [points, wins, goals / n if n else 0, conceded / n if n else 0, n]
    return values, pd.DataFrame(details)


try:
    model, encoder = load_model()
    path = find_data_file()
    if path is None:
        raise FileNotFoundError(f"Dataset introuvable : data/{DATA_FILENAME}")
    data = read_matches(str(path), path.stat().st_mtime)
    if data.empty:
        raise ValueError("Aucun match terminé valide dans le fichier CSV.")
except Exception as exc:
    st.error(f"Impossible de lancer l'application : {exc}")
    st.stop()

# Préférer les équipes de la saison la plus récente quand le CSV possède la colonne Season.
if "Season" in data.columns and data["Season"].notna().any():
    season = str(data["Season"].dropna().iloc[-1])
    current_games = data[data["Season"].astype(str) == season]
else:
    season = "historique disponible"
    current_games = data

teams = sorted(set(current_games["HomeTeam"]) | set(current_games["AwayTeam"]))
if len(teams) < 2:
    st.error("Pas assez d'équipes pour réaliser une prédiction.")
    st.stop()

st.caption(
    f"📁 {path.name}  |  Saison : {season}  |  "
    f"{len(data)} matchs terminés  |  Dernière date : {data['Date'].max():%d/%m/%Y}"
)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("🏠 Équipe à domicile")
    home_team = st.selectbox("Choisir l'équipe à domicile", teams, index=0)
    home_stats, home_details = recent_form(data, home_team)
    st.metric("Points des 5 derniers matchs", f"{home_stats[0]:.0f}/15")
    st.caption(f"Victoires : {home_stats[1]:.0f} · Buts/match : {home_stats[2]:.2f} · Encaissés/match : {home_stats[3]:.2f}")
    with st.expander("Voir ses derniers matchs"):
        st.dataframe(home_details, use_container_width=True, hide_index=True)

with right:
    st.subheader("✈️ Équipe à l'extérieur")
    away_team = st.selectbox("Choisir l'équipe à l'extérieur", teams, index=1)
    away_stats, away_details = recent_form(data, away_team)
    st.metric("Points des 5 derniers matchs", f"{away_stats[0]:.0f}/15")
    st.caption(f"Victoires : {away_stats[1]:.0f} · Buts/match : {away_stats[2]:.2f} · Encaissés/match : {away_stats[3]:.2f}")
    with st.expander("Voir ses derniers matchs"):
        st.dataframe(away_details, use_container_width=True, hide_index=True)

if home_team == away_team:
    st.warning("Choisis deux équipes différentes.")
else:
    if st.button("🔮 Générer la prédiction", type="primary", use_container_width=True):
        X_new = pd.DataFrame([home_stats + away_stats], columns=FEATURES)
        try:
            probabilities = model.predict_proba(X_new)[0]
            # Ne pas supposer l'ordre H/D/A: les classes sont encodées A=0,D=1,H=2.
            predicted_labels = encoder.inverse_transform(model.classes_.astype(int))
            results = {label: float(prob) * 100 for label, prob in zip(predicted_labels, probabilities)}
            if set(results) != {"H", "D", "A"}:
                raise ValueError(f"Classes inattendues : {list(results)}")
        except Exception as exc:
            st.error(f"Erreur de prédiction : {exc}")
            st.stop()

        st.markdown(f"""
        <div class="prob-dashboard">
           <div class="prob-box box-home">
             <div class="prob-title">Victoire {escape(home_team)}</div>
             <div class="prob-score">{results['H']:.1f}%</div>
           </div>
           <div class="prob-box box-draw">
             <div class="prob-title">Match nul</div>
             <div class="prob-score">{results['D']:.1f}%</div>
           </div>
           <div class="prob-box box-away">
             <div class="prob-title">Victoire {escape(away_team)}</div>
             <div class="prob-score">{results['A']:.1f}%</div>
           </div>
        </div>
        """, unsafe_allow_html=True)
        