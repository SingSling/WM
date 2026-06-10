"""Spieler-Qualitätsscore aus Elo + EAR-180.

Pipeline:

1. Eingabe ist ``Data/player_elo_values.csv`` (vom Elo-Joiner erzeugt).
2. Pro Kicker-Position werden ``Elo`` und ``EAR-180`` jeweils
   z-standardisiert: ``z = (x − μ_pos) / σ_pos``. Position statt Team,
   damit Torhüter nicht systematisch unter Stürmern leiden.
   Spieler ohne EAR-Lesart bekommen ``z_EAR = 0`` (positions-neutral).
3. ``Qualität (roh) = 0.5 · z_Elo + 0.5 · z_EAR``.
4. Innerhalb jedes Teams pro Position wird gerankt. Top-3
   Feldspieler bleiben, Reservisten bekommen ``× 0.5``. Bei
   Torhütern bleibt nur der beste; Backup-GKs werden halbiert.

Negative Roh-Scores werden mit dem 0.5-Faktor näher an 0 gebracht
(weniger negativ) — bewusst akzeptiert, dies entspricht der wörtlichen
„reduce by 50 %"-Auslegung. Wer das anders haben möchte, dreht am
``PENALTY``-Faktor oder klemmt zuerst auf ≥ 0.

Ausgabe: ``Data/player_quality.csv`` mit ID + Komponenten + Endwert.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
ELO_MATCHES_PATH = DATA_DIR / "player_elo_values.csv"
KICKER_PATH = DATA_DIR / "players-se-k01012026.csv"
OUT_PATH = DATA_DIR / "player_quality.csv"


# Wie viele Spieler einer Position auf einem Team behalten ihren Roh-Score?
# Außerhalb: ``× PENALTY``.
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

    df["Qualität (roh)"] = 0.5 * df["z_Elo"] + 0.5 * df["z_EAR"]

    # Pro Team × Position ranken (1 = bester).
    df["Pos-Rang im Team"] = (
        df.groupby(["Verein", "Position"])["Qualität (roh)"]
          .rank(ascending=False, method="first")
          .astype(int)
    )

    keep_thresholds = df["Position"].map(KEEP_TOP_BY_POS)
    df["Qualität"] = df["Qualität (roh)"].where(
        df["Pos-Rang im Team"] <= keep_thresholds,
        df["Qualität (roh)"] * PENALTY,
    )
    return df


def write_csv(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    cols = [
        "ID", "Verein", "Position",
        "Elo", "EAR-180", "z_Elo", "z_EAR",
        "Qualität (roh)", "Pos-Rang im Team", "Qualität",
    ]
    out = df[cols].copy()
    for c in ("z_Elo", "z_EAR", "Qualität (roh)", "Qualität"):
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

    cols = [
        "Angezeigter Name", "Verein", "Position",
        "z_Elo", "z_EAR", "Qualität (roh)",
        "Pos-Rang im Team", "Qualität",
    ]
    print("Top 15 nach Qualität:")
    print(show.sort_values("Qualität", ascending=False).head(15)[cols].to_string(index=False))
    print("\nBeispiel: schlechtester gerankter Top-3-Stürmer (Vergleich Roh vs. Endwert):")
    sample = show[(show["Position"] == "FORWARD") & (show["Pos-Rang im Team"] == 3)]
    print(sample.sort_values("Qualität", ascending=False).head(5)[cols].to_string(index=False))
    print("\nBeispiel: erster halbierter Backup-Stürmer pro Top-Team:")
    backup = show[(show["Position"] == "FORWARD") & (show["Pos-Rang im Team"] == 4)]
    print(backup.sort_values("Qualität (roh)", ascending=False).head(5)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
