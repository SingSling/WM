"""Erwartete Spiele pro Spieler.

Verknüpft zwei Signale zu einer Spieler-Kennzahl, die direkt in den
Optimizer einfließen kann:

* ``Data/lineup_probabilities.csv`` — Startelf-Wahrscheinlichkeit pro
  Spieler (aus :mod:`lineup_probabilities`).
* Simulator-Ausgabe — ``exp_games`` pro Team (aus :func:`simulator.run_monte_carlo`).

Annahme: ein Spieler "spielt" ein Turnierspiel mit seiner Startelf-
Wahrscheinlichkeit. Erwartete Spiele = ``probability × team.exp_games``.

Ausgabe: ``Data/expected_player_games.csv`` mit allen CSV-Spielern und
einer Spalte ``Erwartete Spiele`` (Float, 3 Nachkommastellen). Spieler
ohne Lineup-Daten erhalten Probability 0 → Erwartete Spiele 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ratings import load_ratings
from schedule import load_schedule
from simulator import run_monte_carlo


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PLAYERS_PATH = DATA_DIR / "players-se-k01012026.csv"
LINEUP_PROB_PATH = DATA_DIR / "lineup_probabilities.csv"
PLAYER_QUALITY_PATH = DATA_DIR / "player_quality.csv"
OUT_PATH = DATA_DIR / "expected_player_games.csv"


# Simulator-/Ratings-Teamnamen (Englisch) -> CSV-"Verein"-Spalte (Deutsch).
# Hinweis: Eine ähnliche Tabelle gibt es in `lineup_probabilities.TEAM_EN_TO_DE`,
# aber die Quelle dort (Lineup-JSON) nutzt andere EN-Schreibweisen
# ("Czech Republic" statt "Czechia", "Bosnia" statt "Bosnia-Herzegovina",
# "DR Congo" statt "Congo DR"). Daher ein eigener Mapping-Layer hier.
SIM_TEAM_TO_DE: dict[str, str] = {
    "Mexico": "Mexiko",
    "South Africa": "Südafrika",
    "South Korea": "Südkorea",
    "Czechia": "Tschechien",
    "Canada": "Kanada",
    "Bosnia-Herzegovina": "Bosnien-Herzegowina",
    "Qatar": "Katar",
    "Switzerland": "Schweiz",
    "Brazil": "Brasilien",
    "Morocco": "Marokko",
    "Haiti": "Haiti",
    "Scotland": "Schottland",
    "USA": "USA",
    "Paraguay": "Paraguay",
    "Australia": "Australien",
    "Turkey": "Türkei",
    "Germany": "Deutschland",
    "Curacao": "Curacao",
    "Ivory Coast": "Elfenbeinküste",
    "Ecuador": "Ecuador",
    "Netherlands": "Niederlande",
    "Japan": "Japan",
    "Sweden": "Schweden",
    "Tunisia": "Tunesien",
    "Belgium": "Belgien",
    "Egypt": "Ägypten",
    "Iran": "Iran",
    "New Zealand": "Neuseeland",
    "Spain": "Spanien",
    "Cape Verde": "Kap Verde",
    "Saudi Arabia": "Saudi-Arabien",
    "Uruguay": "Uruguay",
    "France": "Frankreich",
    "Senegal": "Senegal",
    "Iraq": "Irak",
    "Norway": "Norwegen",
    "Argentina": "Argentinien",
    "Algeria": "Algerien",
    "Austria": "Österreich",
    "Jordan": "Jordanien",
    "Portugal": "Portugal",
    "Congo DR": "DR Kongo",
    "Uzbekistan": "Usbekistan",
    "Colombia": "Kolumbien",
    "England": "England",
    "Croatia": "Kroatien",
    "Ghana": "Ghana",
    "Panama": "Panama",
}


# Welche Team-Metriken aus dem Monte-Carlo-Output unterstützen wir und welche
# Spalte produzieren sie auf Spielerebene? Erweiterbar (für weitere
# Erwartungswerte wäre hier zu ergänzen).
METRIC_SPECS: dict[str, dict[str, str]] = {
    "exp_games": {"player_col": "Erwartete Spiele"},
    "exp_gd":    {"player_col": "Erwartete Tordifferenz"},
}


def mc_to_team_metric_de(mc: pd.DataFrame, metric_col: str) -> dict[str, float]:
    """Wandelt eine Simulator-Spalte in {team_de: value} um.

    Übersetzt die EN-Teamnamen aus der Simulator-Ausgabe in das in der
    Spieler-CSV verwendete Deutsch.
    """
    missing = set(mc["team"]) - set(SIM_TEAM_TO_DE)
    if missing:
        raise KeyError(f"Kein DE-Mapping für Simulator-Teams: {sorted(missing)}")
    if metric_col not in mc.columns:
        raise KeyError(f"Spalte '{metric_col}' fehlt im Monte-Carlo-Output")
    return {
        SIM_TEAM_TO_DE[r["team"]]: float(r[metric_col])
        for r in mc[["team", metric_col]].to_dict("records")
    }


def mc_to_team_exp_games_de(mc: pd.DataFrame) -> dict[str, float]:
    """Kurzform für ``mc_to_team_metric_de(mc, 'exp_games')`` (back-compat)."""
    return mc_to_team_metric_de(mc, "exp_games")


def load_default_probabilities() -> dict[str, float]:
    """Lädt die Default-Startelf-Wahrscheinlichkeiten aus der CSV ({ID: p}).

    Quelle für die *vorberechnete* ``expected_player_games.csv``. Das
    Dashboard nutzt stattdessen :func:`load_default_quality_multipliers`
    als Anfangswert für die editierbaren Slider.
    """
    df = pd.read_csv(LINEUP_PROB_PATH, sep=";")[["ID", "Probability"]]
    # Upstream-Daten können vereinzelt Probability > 1 liefern (z.B. wenn ein
    # Spieler in einer Quelle mehrfach gematcht wird). Klemmen, damit
    # "Erwartete Spiele" semantisch ≤ exp_games bleibt.
    df["Probability"] = df["Probability"].clip(lower=0.0, upper=1.0)
    return dict(zip(df["ID"], df["Probability"]))


def load_default_quality_multipliers() -> dict[str, float]:
    """Qualitäts-Multiplikator ({ID: m}) als Default für das Dashboard-Slider-UI.

    Liest ``Data/player_quality.csv`` (vom Quality-Pipeline-Skript). Fehlt
    die Datei, fällt die Funktion auf die Lineup-Wahrscheinlichkeiten
    zurück, damit das Dashboard trotzdem bootet.
    """
    if not PLAYER_QUALITY_PATH.exists():
        return load_default_probabilities()
    df = pd.read_csv(PLAYER_QUALITY_PATH, sep=";")[["ID", "Qualitäts-Multiplikator"]]
    df["Qualitäts-Multiplikator"] = df["Qualitäts-Multiplikator"].clip(0.0, 1.0)
    return dict(zip(df["ID"], df["Qualitäts-Multiplikator"]))


def apply_expected_metric(
    players: pd.DataFrame,
    team_metric: dict[str, float],
    probabilities: dict[str, float],
    out_col: str,
) -> pd.DataFrame:
    """Hängt die Spalte ``out_col`` an einen Spieler-DataFrame.

    ``team_metric`` ist ein Dict ``{Verein_DE: metric}``, ``probabilities``
    ein Dict ``{Spieler-ID: p}``. Resultat = ``p × team_metric``. Spieler
    ohne Eintrag in ``probabilities`` werden mit p=0 behandelt; fehlende
    Vereine lösen einen Fehler aus.
    """
    out = players.copy()
    out["Probability"] = out["ID"].map(probabilities).fillna(0.0).clip(0.0, 1.0)
    tmp_col = f"_{out_col}_team"
    out[tmp_col] = out["Verein"].map(team_metric)

    missing_teams = out.loc[out[tmp_col].isna(), "Verein"].unique()
    if len(missing_teams):
        raise KeyError(
            f"Keine Team-Werte für CSV-Vereine: {sorted(missing_teams)}"
        )
    out[out_col] = out["Probability"] * out[tmp_col]
    return out.drop(columns=tmp_col)


def apply_expected_games(
    players: pd.DataFrame,
    team_exp_games: dict[str, float],
    probabilities: dict[str, float],
) -> pd.DataFrame:
    """Back-compat-Wrapper: setzt nur die Spalte ``Erwartete Spiele``.

    Erhält zusätzlich die Spalte ``exp_games`` (Team-Wert) im Resultat, wie
    in der ursprünglichen Implementierung.
    """
    out = players.copy()
    out["Probability"] = out["ID"].map(probabilities).fillna(0.0).clip(0.0, 1.0)
    out["exp_games"] = out["Verein"].map(team_exp_games)

    missing_teams = out.loc[out["exp_games"].isna(), "Verein"].unique()
    if len(missing_teams):
        raise KeyError(
            f"Keine exp_games für CSV-Vereine: {sorted(missing_teams)}"
        )
    out["Erwartete Spiele"] = out["Probability"] * out["exp_games"]
    return out


def team_expected_games(
    n_runs: int = 10_000, seed: int = 42, source: str = "elo",
) -> pd.DataFrame:
    """Liefert pro Team die erwartete Anzahl Turnierspiele (Index: DE-Name)."""
    schedule = load_schedule()
    ratings = load_ratings(source=source)
    mc = run_monte_carlo(schedule, ratings, n_runs=n_runs, seed=seed)
    return pd.DataFrame.from_dict(
        mc_to_team_exp_games_de(mc), orient="index", columns=["exp_games"]
    )


def build_expected_player_metrics(
    n_runs: int = 10_000, seed: int = 42, source: str = "elo",
) -> pd.DataFrame:
    """Verknüpft Lineup-Wahrscheinlichkeiten mit Team-Erwartungswerten.

    Resultat enthält alle Spieler aus der Stammdaten-CSV inklusive der
    Team-Spalten ``exp_games`` / ``exp_gd`` und der Spielerspalten
    ``Erwartete Spiele`` und ``Erwartete Tordifferenz``.
    """
    schedule = load_schedule()
    ratings = load_ratings(source=source)
    mc = run_monte_carlo(schedule, ratings, n_runs=n_runs, seed=seed)

    players = pd.read_csv(PLAYERS_PATH, sep=";")
    probs = load_default_probabilities()

    # Probability einmal anhängen — danach für jede Metrik (Team-Spalte +
    # Spieler-Spalte) ein Produkt bilden.
    out = players.copy()
    out["Probability"] = out["ID"].map(probs).fillna(0.0).clip(0.0, 1.0)

    for team_col, spec in METRIC_SPECS.items():
        team_vals = mc_to_team_metric_de(mc, team_col)
        out[team_col] = out["Verein"].map(team_vals)
        missing = out.loc[out[team_col].isna(), "Verein"].unique()
        if len(missing):
            raise KeyError(f"Keine {team_col} für CSV-Vereine: {sorted(missing)}")
        out[spec["player_col"]] = out["Probability"] * out[team_col]
    return out


def build_expected_player_games(
    n_runs: int = 10_000, seed: int = 42, source: str = "elo",
) -> pd.DataFrame:
    """Back-compat-Wrapper: liefert nur ``Erwartete Spiele``."""
    return build_expected_player_metrics(n_runs=n_runs, seed=seed, source=source)


def write_csv(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    cols = [
        "ID", "Vorname", "Nachname", "Angezeigter Name", "Verein",
        "Position", "Marktwert", "Probability",
        "exp_games", "Erwartete Spiele",
        "exp_gd", "Erwartete Tordifferenz",
    ]
    out = df[cols].copy()
    for c in cols:
        if c not in {"ID", "Vorname", "Nachname", "Angezeigter Name",
                     "Verein", "Position", "Marktwert"}:
            out[c] = out[c].round(3)
    out.to_csv(path, sep=";", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source", choices=["elo", "betting", "custom"], default="elo",
        help="Rating-Quelle für den Simulator",
    )
    args = parser.parse_args()

    df = build_expected_player_metrics(n_runs=args.runs, seed=args.seed, source=args.source)
    write_csv(df)
    pd.set_option("display.max_colwidth", 30)

    cols = ["Angezeigter Name", "Verein", "Position", "Probability",
            "exp_games", "Erwartete Spiele",
            "exp_gd", "Erwartete Tordifferenz"]
    print(f"\nTop 15 nach Erwarteten Spielen ({args.runs:,} Sim-Läufe):\n")
    print(df.sort_values("Erwartete Spiele", ascending=False).head(15)[cols].to_string(index=False))
    print(f"\nTop 15 nach Erwarteter Tordifferenz:\n")
    print(df.sort_values("Erwartete Tordifferenz", ascending=False).head(15)[cols].to_string(index=False))
    print(f"\nGeschrieben: {OUT_PATH}")


if __name__ == "__main__":
    main()
