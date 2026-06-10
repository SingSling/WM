"""Verknüpft Kicker-Spieler mit Transfermarkt-Marktwerten.

Eingaben:
* ``Data/players_tm.csv`` — Transfermarkt-Stammdaten (Kaggle-Dump).
* ``Data/player_valuations_tm.csv`` — historische Marktwerte (für Fallback,
  falls ``market_value_in_eur`` in den Stammdaten fehlt).
* ``Data/players-se-k01012026.csv`` — Kicker-Stammdaten.

Ausgabe: ``Data/transfermarkt_values.csv`` mit (Kicker-ID, Marktwert TM).
Nicht-gematchte Kicker-Spieler werden in ``Data/tm_unmatched.txt``
protokolliert.

Matching-Strategie:
1. TM-Spieler auf relevante WM-Nationen filtern (Citizenship → DE-Team).
2. Für jeden Kicker-Spieler: TM-Kandidaten gleicher Nation → Namens-Match
   via :func:`lineup_probabilities.match_prediction`.
3. Marktwert = ``market_value_in_eur`` (Stammdaten), Fallback letzter
   Eintrag in ``player_valuations_tm``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lineup_probabilities import (
    match_prediction,
    normalize,
    player_aliases,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
TM_PLAYERS_PATH = DATA_DIR / "players_tm.csv"
TM_VALUATIONS_PATH = DATA_DIR / "player_valuations_tm.csv"
KICKER_PATH = DATA_DIR / "players-se-k01012026.csv"
OUT_PATH = DATA_DIR / "transfermarkt_values.csv"
UNMATCHED_LOG = DATA_DIR / "tm_unmatched.txt"
STRENGTH_PATH = DATA_DIR / "tm_strength.csv"
EXPECTED_GAMES_PATH = DATA_DIR / "expected_player_games.csv"


# TM ``country_of_citizenship`` → kicker ``Verein`` (DE).
# Wir filtern erst auf die 48 WM-Nationen — Spieler ohne TM-Bürgerschaft in
# dieser Liste werden ignoriert (Sondereinbürgerungen werden im
# Namens-Fallback unten trotzdem versucht).
TM_COUNTRY_TO_DE: dict[str, str] = {
    "Mexico": "Mexiko",
    "South Africa": "Südafrika",
    "Korea, South": "Südkorea",
    "Czech Republic": "Tschechien",
    "Canada": "Kanada",
    "Bosnia-Herzegovina": "Bosnien-Herzegowina",
    "Qatar": "Katar",
    "Switzerland": "Schweiz",
    "Brazil": "Brasilien",
    "Morocco": "Marokko",
    "Haiti": "Haiti",
    "Scotland": "Schottland",
    "United States": "USA",
    "Paraguay": "Paraguay",
    "Australia": "Australien",
    "Türkiye": "Türkei",
    "Germany": "Deutschland",
    "Curacao": "Curacao",
    "Cote d'Ivoire": "Elfenbeinküste",
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
    "DR Congo": "DR Kongo",
    "Uzbekistan": "Usbekistan",
    "Colombia": "Kolumbien",
    "England": "England",
    "Croatia": "Kroatien",
    "Ghana": "Ghana",
    "Panama": "Panama",
}


def load_tm_players(min_last_season: int = 2023) -> pd.DataFrame:
    """Lädt Transfermarkt-Spieler, gefiltert auf aktive WM-Nationen."""
    cols = [
        "player_id", "name", "first_name", "last_name",
        "country_of_citizenship", "sub_position", "position",
        "market_value_in_eur", "last_season",
    ]
    df = pd.read_csv(TM_PLAYERS_PATH, usecols=cols)
    # Nur halbwegs aktuelle Spieler.
    df = df[df["last_season"] >= min_last_season].copy()
    # DE-Verein anhängen (für nicht-WM-Länder None → später rausgefiltert).
    df["team_de"] = df["country_of_citizenship"].map(TM_COUNTRY_TO_DE)
    return df.reset_index(drop=True)


# Kicker-Position → TM-``position``-Spalte (grob).
KICKER_POS_TO_TM: dict[str, str] = {
    "GOALKEEPER": "Goalkeeper",
    "DEFENDER":   "Defender",
    "MIDFIELDER": "Midfield",
    "FORWARD":    "Attack",
}


def latest_valuation_per_player() -> pd.Series:
    """{player_id: latest market_value_in_eur} aus dem Valuations-Dump."""
    v = pd.read_csv(
        TM_VALUATIONS_PATH,
        usecols=["player_id", "date", "market_value_in_eur"],
    )
    # Sortieren und je Spieler den jüngsten Eintrag nehmen.
    v = v.sort_values(["player_id", "date"]).drop_duplicates(
        subset="player_id", keep="last"
    )
    return v.set_index("player_id")["market_value_in_eur"]


def _enriched_candidates(tm: pd.DataFrame) -> dict[str, list[dict]]:
    """Pro DE-Team: Liste von Kandidaten mit vorberechneten Aliases."""
    by_team: dict[str, list[dict]] = {}
    for row in tm.itertuples(index=False):
        if pd.isna(row.team_de):
            continue
        first = row.first_name if isinstance(row.first_name, str) else ""
        last = row.last_name if isinstance(row.last_name, str) else ""
        display = row.name if isinstance(row.name, str) else ""
        cand = {
            "tm_id": int(row.player_id),
            "name": display,
            "first": first,
            "last": last,
            "display": display,
            "display_short": last or display,
            "position": row.position if isinstance(row.position, str) else "",
            "market_value": row.market_value_in_eur,
        }
        cand["_aliases"] = player_aliases(first, last, display)
        cand["_aliases"].add(normalize(display))
        by_team.setdefault(row.team_de, []).append(cand)
    return by_team


def _resolve_ambiguous_single_name(
    pred_norm: str, candidates: list[dict],
) -> dict | None:
    """Tiebreaker für Single-Name-Matches: höchster Marktwert gewinnt.

    Nur anwenden, wenn alle Treffer denselben normalisierten Display-Namen
    haben (z.B. fünf brasilianische ``Marquinhos``-Einträge). Sonst lieber
    None zurückgeben statt zu raten.
    """
    same_name = [
        c for c in candidates
        if normalize(c["display"]) == pred_norm
    ]
    if not same_name:
        return None
    # Marktwert kann NaN sein → den höchsten "echten" Wert bevorzugen.
    valued = [c for c in same_name if pd.notna(c.get("market_value"))]
    if valued:
        return max(valued, key=lambda c: c["market_value"])
    # Alle ohne Wert → einer von ihnen, egal welcher.
    return same_name[0]


def build_values() -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Joint Kicker × Transfermarkt und liefert (df, unmatched_rows)."""
    kicker = pd.read_csv(KICKER_PATH, sep=";")
    tm = load_tm_players()
    valuations = latest_valuation_per_player()

    # Marktwert: bevorzugt aus Stammdaten, sonst aus historischen Valuations.
    fallback_vals = tm["player_id"].map(valuations)
    tm["market_value_in_eur"] = tm["market_value_in_eur"].fillna(fallback_vals)

    candidates_by_team = _enriched_candidates(tm)

    matches: list[dict] = []
    unmatched: list[tuple[str, str, str]] = []
    for _, row in kicker.iterrows():
        pid = row["ID"]
        team_de = row["Verein"]
        display = row["Angezeigter Name"]
        kpos = row["Position"]
        tm_pos = KICKER_POS_TO_TM.get(kpos)

        all_cands = candidates_by_team.get(team_de, [])
        # Erst auf Position einschränken — robust gegen mehrere
        # Same-Name-Spieler unterschiedlicher Position.
        cands = [c for c in all_cands if c["position"] == tm_pos] if tm_pos else all_cands

        match = match_prediction(display, cands) if cands else None
        if match is None and cands:
            match = _resolve_ambiguous_single_name(normalize(display), cands)
        # Fallback: gleiche Strategie ohne Positions-Filter (falls TM eine
        # andere Position vermerkt hat als kicker).
        if match is None and all_cands:
            match = match_prediction(display, all_cands)
            if match is None:
                match = _resolve_ambiguous_single_name(normalize(display), all_cands)

        if match is None:
            unmatched.append((team_de, pid, display))
            continue
        mv = match.get("market_value")
        mv_value = None if pd.isna(mv) else int(mv)
        matches.append({
            "ID": pid,
            "Marktwert TM": mv_value,
            "TM-ID": match["tm_id"],
            "TM-Name": match["name"],
            "Verein": team_de,
        })

    df = pd.DataFrame(matches)
    return df, unmatched


def build_strength_table(values_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Leitet Stärke-Scores aus Marktwerten ab und multipliziert mit Team-Werten.

    Stärke = Marktwert / max(Marktwert im selben WM-Team), zwischen 0 und 1.
    Damit ist der teuerste Spieler jedes Teams 1,0, ein halb so wertvoller
    Backup 0,5, ein Spieler ohne TM-Match 0,0.

    Zusätzlich werden Strength × Team-Erwartungswert berechnet:
    * ``Erwartete Spiele (TM)`` = Strength × ``exp_games`` (Team).
    * ``Erwartete Tordifferenz (TM)`` = Strength × ``exp_gd`` (Team).

    Beide nutzen die Team-Werte aus :data:`EXPECTED_GAMES_PATH`. Fehlt die
    Datei, wird sie über ``expected_player_games`` erzeugt.
    """
    if values_df is None:
        if not OUT_PATH.exists():
            raise FileNotFoundError(
                f"{OUT_PATH} fehlt — zuerst `build_values()` laufen lassen."
            )
        values_df = pd.read_csv(OUT_PATH, sep=";")

    # Kicker-Stammdaten für vollständige Spielerliste (auch ohne TM-Match).
    kicker = pd.read_csv(KICKER_PATH, sep=";")[["ID", "Verein", "Position"]]
    df = kicker.merge(
        values_df[["ID", "Marktwert TM"]], on="ID", how="left",
    )

    # Stärke innerhalb des Teams normalisieren.
    team_max = df.groupby("Verein")["Marktwert TM"].transform("max")
    # Falls ein ganzes Team keinen Match hat (sollte nicht passieren), Score = 0.
    df["TM-Stärke"] = (df["Marktwert TM"].fillna(0.0) / team_max).fillna(0.0)
    df["TM-Stärke"] = df["TM-Stärke"].clip(lower=0.0, upper=1.0)

    # Team-Erwartungswerte beschaffen (aus dem Simulator-Output).
    if not EXPECTED_GAMES_PATH.exists():
        # Lazy: gleiche Pipeline wie der Default-Workflow im Optimizer.
        from expected_player_games import build_expected_player_metrics, write_csv
        write_csv(build_expected_player_metrics())
    teams = pd.read_csv(EXPECTED_GAMES_PATH, sep=";")[["ID", "exp_games", "exp_gd"]]
    df = df.merge(teams, on="ID", how="left")

    df["Erwartete Spiele (TM)"] = df["TM-Stärke"] * df["exp_games"]
    df["Erwartete Tordifferenz (TM)"] = df["TM-Stärke"] * df["exp_gd"]
    return df


def write_strength_csv(strength: pd.DataFrame, path: Path = STRENGTH_PATH) -> None:
    cols = [
        "ID", "Verein", "Position",
        "Marktwert TM", "TM-Stärke",
        "exp_games", "Erwartete Spiele (TM)",
        "exp_gd", "Erwartete Tordifferenz (TM)",
    ]
    out = strength[cols].copy()
    for c in ("Marktwert TM",):
        if c in out.columns:
            out[c] = out[c].astype("Int64")  # NaN-tolerant int
    for c in ("TM-Stärke", "exp_games", "exp_gd",
              "Erwartete Spiele (TM)", "Erwartete Tordifferenz (TM)"):
        out[c] = out[c].round(4)
    out.to_csv(path, sep=";", index=False)


def write_outputs(df: pd.DataFrame, unmatched: list[tuple[str, str, str]]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, sep=";", index=False)
    with UNMATCHED_LOG.open("w", encoding="utf-8") as f:
        f.write(f"# Unmatched kicker players: {len(unmatched)}\n")
        for team, pid, name in unmatched:
            f.write(f"{team} | {pid} | {name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-season", type=int, default=2023,
        help="TM-Spieler nur ab dieser last_season berücksichtigen",
    )
    args = parser.parse_args()
    # min_season ist in load_tm_players() hartkodiert; CLI-Flag dient
    # Reproduzierbarkeit für spätere Optionen.
    _ = args  # placeholder

    df, unmatched = build_values()
    write_outputs(df, unmatched)

    total_kicker = len(pd.read_csv(KICKER_PATH, sep=";"))
    matched = len(df)
    with_value = df["Marktwert TM"].notna().sum()
    print(f"Kicker-Spieler gesamt: {total_kicker}")
    print(f"Davon gematcht:        {matched}  ({matched / total_kicker:.1%})")
    print(f"  davon mit Marktwert: {with_value}")
    print(f"Unmatched: {len(unmatched)} → {UNMATCHED_LOG}")
    print(f"Ausgabe:   {OUT_PATH}")

    if with_value:
        print("\nTop 10 nach Marktwert TM (gematcht):")
        top = df.dropna(subset=["Marktwert TM"]).sort_values("Marktwert TM", ascending=False).head(10)
        print(top[["TM-Name", "Verein", "Marktwert TM"]].to_string(index=False))

    # Strength-Tabelle: Marktwert → Score 0..1 (team-normalisiert) und
    # Strength × Team-Erwartungswerte.
    print("\nBerechne TM-Stärke …")
    strength = build_strength_table(df)
    write_strength_csv(strength)
    print(f"Stärke-Output: {STRENGTH_PATH}")

    print("\nTop 15 nach Erwartete Spiele (TM):")
    cols = ["ID", "Verein", "Position", "TM-Stärke",
            "Erwartete Spiele (TM)", "Erwartete Tordifferenz (TM)"]
    top_s = strength.sort_values("Erwartete Spiele (TM)", ascending=False).head(15)
    # Schöne Anzeige: Namen aus Kicker-Stammdaten holen
    names = pd.read_csv(KICKER_PATH, sep=";")[["ID", "Angezeigter Name"]]
    print(top_s.merge(names, on="ID")[
        ["Angezeigter Name", "Verein", "Position", "TM-Stärke",
         "Erwartete Spiele (TM)", "Erwartete Tordifferenz (TM)"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
