"""Verknüpft Kicker-Spieler mit clubelo/global-Elo-Werten.

Eingabe:
* ``Data/players_elo.csv`` — Spieler-Elo (Kaggle / öffentlich verfügbarer Dump).
* ``Data/players-se-k01012026.csv`` — Kicker-Stammdaten.
* ``Data/expected_player_games.csv`` — Team-Erwartungswerte (für die
  Strength-basierten Spalten; wird via :mod:`expected_player_games`
  erzeugt falls nicht vorhanden).

Ausgaben:
* ``Data/player_elo_values.csv`` — pro Kicker-ID: Elo, Rang, TM-Name.
* ``Data/elo_strength.csv`` — analog zu ``tm_strength.csv``: Stärke =
  ``Elo / max(Elo im Team)``; multipliziert mit Team-Werten.
* ``Data/elo_unmatched.txt`` — Audit-Log.

Matching: pro Kicker-Spieler suchen wir TM-Kandidaten mit passender
Nation und Position (Elo-Vokabular → DE-Team, kicker Position →
Elo-Positions-Bucket); der eigentliche Namens-Match nutzt
:func:`lineup_probabilities.match_prediction`. Bei Mehrdeutigkeit
(z.B. mehrere Brasilianer mit Spitznamen ``Marquinhos``) gewinnt der
Spieler mit höchstem Elo.
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
ELO_PATH = DATA_DIR / "players_elo.csv"
KICKER_PATH = DATA_DIR / "players-se-k01012026.csv"
EXPECTED_GAMES_PATH = DATA_DIR / "expected_player_games.csv"
OUT_PATH = DATA_DIR / "player_elo_values.csv"
STRENGTH_PATH = DATA_DIR / "elo_strength.csv"
UNMATCHED_LOG = DATA_DIR / "elo_unmatched.txt"


# Elo ``nationality`` → kicker ``Verein`` (DE).
ELO_COUNTRY_TO_DE: dict[str, str] = {
    "Mexico": "Mexiko",
    "South Africa": "Südafrika",
    "Korea Republic": "Südkorea",
    "Czechia": "Tschechien",
    "Canada": "Kanada",
    "Bosnia and Herzegovina": "Bosnien-Herzegowina",
    "Qatar": "Katar",
    "Switzerland": "Schweiz",
    "Brazil": "Brasilien",
    "Morocco": "Marokko",
    "Haiti": "Haiti",
    "Scotland": "Schottland",
    "USA": "USA",
    "Paraguay": "Paraguay",
    "Australia": "Australien",
    "Türkiye": "Türkei",
    "Germany": "Deutschland",
    "Curaçao": "Curacao",
    "Côte d'Ivoire": "Elfenbeinküste",
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
    "Congo DR": "DR Kongo",  # zweite Schreibweise im Datensatz
    "Uzbekistan": "Usbekistan",
    "Colombia": "Kolumbien",
    "England": "England",
    "Croatia": "Kroatien",
    "Ghana": "Ghana",
    "Panama": "Panama",
}


# Kicker-Position → Elo-``position``-Bucket. ``Forward`` (selten, 13
# Treffer) wird neben ``Attacker`` als Sturm gewertet.
KICKER_POS_TO_ELO: dict[str, set[str]] = {
    "GOALKEEPER": {"Goalkeeper"},
    "DEFENDER":   {"Defender"},
    "MIDFIELDER": {"Midfielder"},
    "FORWARD":    {"Attacker", "Forward"},
}


def load_elo_players() -> pd.DataFrame:
    cols = [
        "player_id", "player_name", "elo", "current_rank",
        "position", "nationality",
    ]
    df = pd.read_csv(ELO_PATH, usecols=cols)
    df["team_de"] = df["nationality"].map(ELO_COUNTRY_TO_DE)
    return df.reset_index(drop=True)


def _enriched_candidates(elo: pd.DataFrame) -> dict[str, list[dict]]:
    """Pro DE-Team: Liste von Kandidaten mit vorberechneten Aliases.

    Der Elo-Datensatz liefert nur einen Vollnamen — wir splitten ihn am
    ersten Leerzeichen in (Vorname, Nachname) für die Match-Heuristik.
    """
    by_team: dict[str, list[dict]] = {}
    for row in elo.itertuples(index=False):
        if pd.isna(row.team_de):
            continue
        full = row.player_name if isinstance(row.player_name, str) else ""
        first, _, last = full.partition(" ")
        if not last:
            last = first  # Single-Name Spieler („Casemiro“)
        cand = {
            "elo_id": int(row.player_id),
            "name": full,
            "first": first,
            "last": last,
            "display": full,
            "display_short": last or full,
            "position": row.position if isinstance(row.position, str) else "",
            "elo": float(row.elo),
            "current_rank": int(row.current_rank) if pd.notna(row.current_rank) else None,
        }
        cand["_aliases"] = player_aliases(first, last, full)
        cand["_aliases"].add(normalize(full))
        by_team.setdefault(row.team_de, []).append(cand)
    return by_team


def _resolve_ambiguous_single_name(
    pred_norm: str, candidates: list[dict],
) -> dict | None:
    """Tiebreaker bei Same-Name-Spielern: höchster Elo gewinnt."""
    same_name = [
        c for c in candidates if normalize(c["display"]) == pred_norm
    ]
    if not same_name:
        return None
    return max(same_name, key=lambda c: c["elo"])


def build_matches() -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    kicker = pd.read_csv(KICKER_PATH, sep=";")
    elo = load_elo_players()
    candidates_by_team = _enriched_candidates(elo)

    rows: list[dict] = []
    unmatched: list[tuple[str, str, str]] = []
    for _, krow in kicker.iterrows():
        pid = krow["ID"]
        team_de = krow["Verein"]
        display = krow["Angezeigter Name"]
        kpos = krow["Position"]
        elo_buckets = KICKER_POS_TO_ELO.get(kpos, set())

        all_cands = candidates_by_team.get(team_de, [])
        # Erst auf Position einschränken (Elo "Unknown" als joker zulassen).
        if elo_buckets:
            cands = [
                c for c in all_cands
                if c["position"] in elo_buckets or c["position"] in ("", "Unknown")
            ]
        else:
            cands = all_cands

        match = match_prediction(display, cands) if cands else None
        if match is None and cands:
            match = _resolve_ambiguous_single_name(normalize(display), cands)
        # Position-Fallback ohne Filter, falls Elo den Spieler abweichend
        # einordnet (z.B. ein RB als "Midfielder").
        if match is None and all_cands:
            match = match_prediction(display, all_cands)
            if match is None:
                match = _resolve_ambiguous_single_name(normalize(display), all_cands)

        if match is None:
            unmatched.append((team_de, pid, display))
            continue
        rows.append({
            "ID": pid,
            "Verein": team_de,
            "Position": kpos,
            "Elo": float(match["elo"]),
            "Elo-Rang": match["current_rank"],
            "Elo-ID": match["elo_id"],
            "Elo-Name": match["name"],
        })
    return pd.DataFrame(rows), unmatched


def build_strength_table(matches_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Elo-Stärke (= Elo / max(Elo im Team)) und Team-multiplizierte Werte."""
    if matches_df is None:
        if not OUT_PATH.exists():
            raise FileNotFoundError(
                f"{OUT_PATH} fehlt — zuerst `build_matches()` laufen lassen."
            )
        matches_df = pd.read_csv(OUT_PATH, sep=";")

    kicker = pd.read_csv(KICKER_PATH, sep=";")[["ID", "Verein", "Position"]]
    df = kicker.merge(matches_df[["ID", "Elo"]], on="ID", how="left")

    team_max = df.groupby("Verein")["Elo"].transform("max")
    df["Elo-Stärke"] = (df["Elo"].fillna(0.0) / team_max).fillna(0.0)
    df["Elo-Stärke"] = df["Elo-Stärke"].clip(lower=0.0, upper=1.0)

    if not EXPECTED_GAMES_PATH.exists():
        from expected_player_games import build_expected_player_metrics, write_csv
        write_csv(build_expected_player_metrics())
    teams = pd.read_csv(EXPECTED_GAMES_PATH, sep=";")[["ID", "exp_games", "exp_gd"]]
    df = df.merge(teams, on="ID", how="left")

    df["Erwartete Spiele (Elo)"] = df["Elo-Stärke"] * df["exp_games"]
    df["Erwartete Tordifferenz (Elo)"] = df["Elo-Stärke"] * df["exp_gd"]
    return df


def write_match_csv(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    out = df.copy()
    out["Elo"] = out["Elo"].round(1)
    out.to_csv(path, sep=";", index=False)


def write_strength_csv(df: pd.DataFrame, path: Path = STRENGTH_PATH) -> None:
    cols = [
        "ID", "Verein", "Position",
        "Elo", "Elo-Stärke",
        "exp_games", "Erwartete Spiele (Elo)",
        "exp_gd", "Erwartete Tordifferenz (Elo)",
    ]
    out = df[cols].copy()
    for c in ("Elo-Stärke", "exp_games", "exp_gd",
              "Erwartete Spiele (Elo)", "Erwartete Tordifferenz (Elo)"):
        out[c] = out[c].round(4)
    out["Elo"] = out["Elo"].round(1)
    out.to_csv(path, sep=";", index=False)


def write_unmatched(unmatched: list[tuple[str, str, str]]) -> None:
    with UNMATCHED_LOG.open("w", encoding="utf-8") as f:
        f.write(f"# Unmatched kicker players: {len(unmatched)}\n")
        for team, pid, name in unmatched:
            f.write(f"{team} | {pid} | {name}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    df, unmatched = build_matches()
    write_match_csv(df)
    write_unmatched(unmatched)

    total_kicker = len(pd.read_csv(KICKER_PATH, sep=";"))
    matched = len(df)
    print(f"Kicker-Spieler gesamt: {total_kicker}")
    print(f"Davon Elo-gematcht:    {matched}  ({matched / total_kicker:.1%})")
    print(f"Unmatched: {len(unmatched)} → {UNMATCHED_LOG}")
    print(f"Match-CSV: {OUT_PATH}")

    if matched:
        print("\nTop 10 nach Elo (gematcht):")
        top = df.sort_values("Elo", ascending=False).head(10)
        print(top[["Elo-Name", "Verein", "Position", "Elo", "Elo-Rang"]].to_string(index=False))

    print("\nBerechne Elo-Stärke …")
    strength = build_strength_table(df)
    write_strength_csv(strength)
    print(f"Stärke-Output: {STRENGTH_PATH}")

    print("\nTop 15 nach Erwartete Spiele (Elo):")
    cols = ["ID", "Verein", "Position", "Elo-Stärke",
            "Erwartete Spiele (Elo)", "Erwartete Tordifferenz (Elo)"]
    names = pd.read_csv(KICKER_PATH, sep=";")[["ID", "Angezeigter Name"]]
    top_s = strength.sort_values("Erwartete Spiele (Elo)", ascending=False).head(15)
    print(top_s.merge(names, on="ID")[
        ["Angezeigter Name", "Verein", "Position", "Elo-Stärke",
         "Erwartete Spiele (Elo)", "Erwartete Tordifferenz (Elo)"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
