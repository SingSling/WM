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


def mc_to_team_exp_games_de(mc: pd.DataFrame) -> dict[str, float]:
    """Wandelt eine Simulator-Ausgabe in {team_de: exp_games} um.

    Übersetzt die EN-Teamnamen aus der Simulator-Ausgabe in das in der
    Spieler-CSV verwendete Deutsch.
    """
    missing = set(mc["team"]) - set(SIM_TEAM_TO_DE)
    if missing:
        raise KeyError(f"Kein DE-Mapping für Simulator-Teams: {sorted(missing)}")
    return {SIM_TEAM_TO_DE[r.team]: float(r.exp_games) for r in mc.itertuples()}


def load_default_probabilities() -> dict[str, float]:
    """Lädt die Default-Startelf-Wahrscheinlichkeiten aus der CSV ({ID: p})."""
    df = pd.read_csv(LINEUP_PROB_PATH, sep=";")[["ID", "Probability"]]
    # Upstream-Daten können vereinzelt Probability > 1 liefern (z.B. wenn ein
    # Spieler in einer Quelle mehrfach gematcht wird). Klemmen, damit
    # "Erwartete Spiele" semantisch ≤ exp_games bleibt.
    df["Probability"] = df["Probability"].clip(lower=0.0, upper=1.0)
    return dict(zip(df["ID"], df["Probability"]))


def apply_expected_games(
    players: pd.DataFrame,
    team_exp_games: dict[str, float],
    probabilities: dict[str, float],
) -> pd.DataFrame:
    """Hängt die Spalte ``Erwartete Spiele`` an einen Spieler-DataFrame.

    ``team_exp_games`` ist ein Dict ``{Verein_DE: exp_games}``,
    ``probabilities`` ein Dict ``{Spieler-ID: p}``. Spieler ohne Eintrag in
    ``probabilities`` werden mit p=0 behandelt; fehlende Vereine in
    ``team_exp_games`` lösen einen Fehler aus.
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
    """Liefert pro Team die erwartete Anzahl Turnierspiele (Index: DE-Name).

    Spalten: ``exp_games``. Übersetzung der Simulator-EN-Namen ins
    Deutsche via :data:`SIM_TEAM_TO_DE`.
    """
    schedule = load_schedule()
    ratings = load_ratings(source=source)
    mc = run_monte_carlo(schedule, ratings, n_runs=n_runs, seed=seed)
    return pd.DataFrame.from_dict(
        mc_to_team_exp_games_de(mc), orient="index", columns=["exp_games"]
    )


def build_expected_player_games(
    n_runs: int = 10_000, seed: int = 42, source: str = "elo",
) -> pd.DataFrame:
    """Verknüpft Lineup-Wahrscheinlichkeiten mit Team-Erwartungswerten.

    Rückgabe enthält alle Spieler aus der Stammdaten-CSV inklusive einer
    Spalte ``Erwartete Spiele`` (= ``Probability × team.exp_games``).
    """
    players = pd.read_csv(PLAYERS_PATH, sep=";")
    team_eg = team_expected_games(n_runs=n_runs, seed=seed, source=source)
    team_dict = team_eg["exp_games"].to_dict()
    return apply_expected_games(players, team_dict, load_default_probabilities())


def write_csv(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    cols = [
        "ID", "Vorname", "Nachname", "Angezeigter Name", "Verein",
        "Position", "Marktwert", "Probability", "exp_games",
        "Erwartete Spiele",
    ]
    out = df[cols].copy()
    out["Probability"] = out["Probability"].round(3)
    out["exp_games"] = out["exp_games"].round(3)
    out["Erwartete Spiele"] = out["Erwartete Spiele"].round(3)
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

    df = build_expected_player_games(n_runs=args.runs, seed=args.seed, source=args.source)
    write_csv(df)

    top = df.sort_values("Erwartete Spiele", ascending=False).head(20)
    cols = ["Angezeigter Name", "Verein", "Position", "Probability",
            "exp_games", "Erwartete Spiele"]
    pd.set_option("display.max_colwidth", 30)
    print(f"\nTop 20 nach Erwarteten Spielen ({args.runs:,} Sim-Läufe):\n")
    print(top[cols].to_string(index=False))
    print(f"\nGeschrieben: {OUT_PATH}")


if __name__ == "__main__":
    main()
