"""Football Match Predictor API - run with: uvicorn api:app --reload"""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
DATA_NAME = "football_2025_2027.csv"
EXPECTED_FEATURES = [
    "Home_Form", "Home_Wins", "Home_Goals_Avg", "Home_Conceded_Avg",
    "Home_Matches_Played", "Away_Form", "Away_Wins", "Away_Goals_Avg",
    "Away_Conceded_Avg", "Away_Matches_Played",
]

app = FastAPI(title="Football Match Predictor API", version="1.0.0")


class MatchRequest(BaseModel):
    home_team: str = Field(min_length=1, examples=["Arsenal"])
    away_team: str = Field(min_length=1, examples=["Liverpool"])


@lru_cache(maxsize=1)
def load_assets():
    model_dir = Path(os.environ.get("MODEL_DIR", str(BASE_DIR / "models")))
    model = joblib.load(model_dir / "football_model.pkl")
    encoder = joblib.load(model_dir / "label_encoder.pkl")
    feature_file = model_dir / "feature_columns.pkl"
    feature_cols = joblib.load(feature_file) if feature_file.exists() else EXPECTED_FEATURES
    if list(feature_cols) != EXPECTED_FEATURES:
        raise ValueError("Les features du modèle ne correspondent pas aux 10 attendues")
    if getattr(model, "n_features_in_", 10) != 10:
        raise ValueError("Le modèle n'utilise pas 10 features. Relancer train_model.py")
    return model, encoder


@lru_cache(maxsize=1)
def load_matches():
    candidates = [
        Path(os.environ["DATA_PATH"]) if os.environ.get("DATA_PATH") else None,
        Path("/data") / DATA_NAME,
        BASE_DIR.parent / "data" / DATA_NAME,
        BASE_DIR / "data" / DATA_NAME,
        BASE_DIR / DATA_NAME,
    ]
    path = next((p for p in candidates if p is not None and p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Dataset introuvable: {DATA_NAME} (ou /data/{DATA_NAME})")
    df = pd.read_csv(path, low_memory=False)
    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes du CSV: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    for col in ("FTHG", "FTAG"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["FTR"] = df["FTR"].astype("string").str.strip().str.upper()
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    df = df.dropna(subset=required)
    df = df.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    df = df.sort_values("Date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Aucun match terminé valide")
    return df


def last_five_stats(df, team):
    recent = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)].tail(5)
    n = len(recent)
    points, wins, goals, conceded = 0, 0, 0.0, 0.0
    for _, match in recent.iterrows():
        is_home = match["HomeTeam"] == team
        won = match["FTR"] == ("H" if is_home else "A")
        drawn = match["FTR"] == "D"
        points += 3 if won else 1 if drawn else 0
        wins += int(won)
        goals += float(match["FTHG"] if is_home else match["FTAG"])
        conceded += float(match["FTAG"] if is_home else match["FTHG"])
    return [points, wins, goals / n if n else 0, conceded / n if n else 0, n]


def get_teams(df):
    if "Season" in df.columns and df["Season"].notna().any():
        latest_season = df["Season"].dropna().iloc[-1]
        df = df[df["Season"] == latest_season]
    return sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))


@app.get("/health")
def health():
    try:
        load_assets()
        load_matches()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok"}


@app.get("/teams")
def teams():
    try:
        return {"teams": get_teams(load_matches())}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: MatchRequest):
    if request.home_team == request.away_team:
        raise HTTPException(status_code=422, detail="Choisir deux équipes différentes")
    try:
        df = load_matches()
        model, encoder = load_assets()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    known_teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
    unknown = [t for t in [request.home_team, request.away_team] if t not in known_teams]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Équipe(s) inconnue(s): {', '.join(unknown)}")

    home = last_five_stats(df, request.home_team)
    away = last_five_stats(df, request.away_team)
    X_new = pd.DataFrame([home + away], columns=EXPECTED_FEATURES)
    try:
        probs = model.predict_proba(X_new)[0]
        labels = encoder.inverse_transform(model.classes_.astype(int))
        result = {label: round(float(prob), 4) for label, prob in zip(labels, probs)}
        if set(result) != {"H", "D", "A"}:
            raise ValueError(f"Classes inattendues: {result.keys()}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prédiction impossible: {exc}") from exc
    return {
        "home_team": request.home_team,
        "away_team": request.away_team,
        "home_win": result["H"],
        "draw": result["D"],
        "away_win": result["A"],
        "prediction": max(result, key=result.get),
        "note": "Probabilités estimées, non garanties",
    }
