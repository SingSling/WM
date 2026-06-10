"""Spieler-Qualitätsscore aus Elo + EAR-180 und passender Spielzeit-Multiplikator.

Pipeline:

1. Eingabe: ``Data/player_elo_values.csv`` (Elo-Joiner) und für die
   abgeleiteten Erwartungswerte ``Data/expected_player_games.csv``
   (Team-Erwartungswerte).
2. Pro Kicker-Position werden ``Elo`` und ``EAR-180`` jeweils
   z-standardisiert (über alle 48 Nationen hinweg). Position statt
   Team, damit Torhüter nicht systematisch unter Stürmern leiden.
   Spieler ohne EAR-Lesart bekommen ``z_EAR = 0`` (positions-neutral).
3. ``Qualität = 0.5 · z_Elo + 0.5 · z_EAR`` — **roher Qualitätsscore
   ohne Team-Anpassung**.
4. ``Qualitäts-Multiplikator (Basis)`` — Min-Max-Normierung von
   ``Qualität`` pro Position. Bester der Position = 1.0, schlechtester
   = 0.0. Damit liegt der Multiplikator in [0, 1].
5. ``Qualitäts-Multiplikator`` — Basis × Team-Rang-Faktor. Top-1 GK /
   Top-3 Feldspieler pro Team behalten ihre Basis (Faktor 1.0), alle
   übrigen werden mit ``0.5`` multipliziert.
6. ``Erwartete Spiele (Qualität) = Multiplikator × team.exp_games``
   und analog für die Tordifferenz.

Ausgabe: ``Data/player_quality.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
ELO_MATCHES_PATH = DATA_DIR / "player_elo_values.csv"
EXPECTED_GAMES_PATH = DATA_DIR / "expected_player_games.csv"
KICKER_PATH = DATA_DIR / "players-se-k01012026.csv"
OUT_PATH = DATA_DIR / "player_quality.csv"


# Top-N pro Team-Position, die volle Spielzeit (Multiplikator 1.0) bekommen.
KEEP_TOP_BY_POS: dict[str, int] = {
    "GOALKEEPER": 1,
    "DEFENDER":   3,
    "MIDFIELDER": 3,
    "FORWARD":    3,
}
PENALTY = 0.5


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if not std or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def compute() -> pd.DataFrame:
    if not ELO_MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"{ELO_MATCHES_PATH} fehlt — bitte zuerst `python3 Code/player_elo_values.py`."
        )
    df = pd.read_csv(ELO_MATCHES_PATH, sep=";")
    needed = {"ID", "Verein", "Position", "Elo", "EAR-180"}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Spalten fehlen in {ELO_MATCHES_PATH.name}: {sorted(missing_cols)}. "
            "Regeneriere die Datei via Code/player_elo_values.py."
        )

    # Z-Scores pro Position (über alle 48 Nationen hinweg).
    df["z_Elo"] = df.groupby("Position")["Elo"].transform(_zscore)
    df["z_EAR"] = df.groupby("Position")["EAR-180"].transform(_zscore)
    df["z_EAR"] = df["z_EAR"].fillna(0.0)

    # Roher Qualitätsscore — keine Team-Anpassung.
    df["Qualität"] = 0.5 * df["z_Elo"] + 0.5 * df["z_EAR"]

    # Basis-Multiplikator: Min-Max-Normierung der Qualität pro Position.
    def _min_max(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)
    df["Qualitäts-Multiplikator (Basis)"] = (
        df.groupby("Position")["Qualität"].transform(_min_max)
    )

    # Team × Position-Ranking (1 = bester) für den Penalty-Faktor.
    df["Pos-Rang im Team"] = (
        df.groupby(["Verein", "Position"])["Qualität"]
          .rank(ascending=False, method="first")
          .astype(int)
    )
    keep_thresholds = df["Position"].map(KEEP_TOP_BY_POS)
    # Finaler Multiplikator: Basis behalten (Top-N) oder × PENALTY (Backups).
    df["Qualitäts-Multiplikator"] = df["Qualitäts-Multiplikator (Basis)"].where(
        df["Pos-Rang im Team"] <= keep_thresholds,
        df["Qualitäts-Multiplikator (Basis)"] * PENALTY,
    )

    # Team-Erwartungswerte beimergen für die abgeleiteten Spielerspalten.
    if not EXPECTED_GAMES_PATH.exists():
        from expected_player_games import build_expected_player_metrics, write_csv as _write_expected
        _write_expected(build_expected_player_metrics())
    team_metrics = pd.read_csv(EXPECTED_GAMES_PATH, sep=";")[
        ["ID", "exp_games", "exp_gd"]
    ]
    df = df.merge(team_metrics, on="ID", how="left")

    df["Erwartete Spiele (Qualität)"] = df["Qualitäts-Multiplikator"] * df["exp_games"]
    df["Erwartete Tordifferenz (Qualität)"] = df["Qualitäts-Multiplikator"] * df["exp_gd"]
    return df


def write_csv(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    cols = [
        "ID", "Verein", "Position",
        "Elo", "EAR-180", "z_Elo", "z_EAR", "Qualität",
        "Pos-Rang im Team",
        "Qualitäts-Multiplikator (Basis)", "Qualitäts-Multiplikator",
        "exp_games", "Erwartete Spiele (Qualität)",
        "exp_gd", "Erwartete Tordifferenz (Qualität)",
    ]
    out = df[cols].copy()
    for c in ("z_Elo", "z_EAR", "Qualität",
              "Qualitäts-Multiplikator (Basis)", "Qualitäts-Multiplikator",
              "exp_games", "Erwartete Spiele (Qualität)",
              "exp_gd", "Erwartete Tordifferenz (Qualität)"):
        out[c] = out[c].round(4)
    out["Elo"] = out["Elo"].round(1)
    if "EAR-180" in out.columns:
        out["EAR-180"] = out["EAR-180"].round(3)
    out.to_csv(path, sep=";", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    df = compute()
    write_csv(df)
    names = pd.read_csv(KICKER_PATH, sep=";")[["ID", "Angezeigter Name"]]
    show = df.merge(names, on="ID")

    print(f"Spieler bewertet: {len(df)}")
    print(f"Ausgabe:          {OUT_PATH}\n")

    qcols = [
        "Angezeigter Name", "Verein", "Position",
        "z_Elo", "z_EAR", "Qualität",
        "Pos-Rang im Team", "Qualitäts-Multiplikator",
    ]
    print("Top 15 nach Qualität (ohne Team-Anpassung):")
    print(show.sort_values("Qualität", ascending=False).head(15)[qcols].to_string(index=False))

    ecols = [
        "Angezeigter Name", "Verein", "Position",
        "Qualität", "Qualitäts-Multiplikator",
        "exp_games", "Erwartete Spiele (Qualität)",
    ]
    print("\nTop 15 nach Erwartete Spiele (Qualität):")
    print(show.sort_values("Erwartete Spiele (Qualität)", ascending=False).head(15)[ecols].to_string(index=False))


if __name__ == "__main__":
    main()
