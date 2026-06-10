"""Aggregiert pro WM-Team eine Team-Elo aus den Spieler-Elos.

Formel:

* Bester Torhüter (Top 1) ·············· 10 %
* Mittel der vier besten Verteidiger · 25 %
* Mittel der fünf besten Mittelfeldspieler · 30 %
* Mittel der drei besten Stürmer · 35 %

Eingabe: ``Data/player_elo_values.csv`` (aus ``Code/player_elo_values.py``).
Ausgabe: ``Data/team_elo_from_players.csv`` und (programmatisch) ein
Dict ``{team_en: rating}`` für den Simulator.

Edge Cases:
* Hat ein Team weniger Spieler einer Position als gefordert, wird über
  alle vorhandenen gemittelt.
* Fehlt eine Position komplett, wird ihr Gewicht auf die übrigen
  Positionen proportional umverteilt (statt mit 0 zu rechnen).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# DE → EN Mapping (Simulator-Vokabular) — die Inverse zu SIM_TEAM_TO_DE.
from expected_player_games import SIM_TEAM_TO_DE


DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
ELO_PATH = DATA_DIR / "player_elo_values.csv"
OUT_PATH = DATA_DIR / "team_elo_from_players.csv"


# Positions-Gewichtung und Anzahl Top-Spieler je Position.
POSITION_SPEC: dict[str, dict[str, float | int]] = {
    "GOALKEEPER": {"weight": 0.10, "top_n": 1},
    "DEFENDER":   {"weight": 0.25, "top_n": 4},
    "MIDFIELDER": {"weight": 0.30, "top_n": 5},
    "FORWARD":    {"weight": 0.35, "top_n": 3},
}


def _de_to_en() -> dict[str, str]:
    return {de: en for en, de in SIM_TEAM_TO_DE.items()}


def _position_average(group: pd.DataFrame, top_n: int) -> float | None:
    """Mittel der besten ``top_n`` Elos in ``group``; None wenn leer."""
    if group.empty:
        return None
    top = group.nlargest(top_n, "Elo")
    return float(top["Elo"].mean())


def compute_team_elo(player_elo: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert Spieler-Elos → eine Zeile pro Team (DE)."""
    rows = []
    for team_de, players in player_elo.groupby("Verein"):
        details: dict[str, float | None] = {}
        for pos, spec in POSITION_SPEC.items():
            pos_group = players[players["Position"] == pos]
            details[pos] = _position_average(pos_group, int(spec["top_n"]))

        # Vorhandene Positionen ausweisen; fehlende Gewichte umverteilen.
        present_weights = {
            pos: float(POSITION_SPEC[pos]["weight"])
            for pos in POSITION_SPEC if details[pos] is not None
        }
        if not present_weights:
            continue  # Team ohne irgendeinen Match — sollte praktisch nie passieren.
        weight_sum = sum(present_weights.values())
        normed = {pos: w / weight_sum for pos, w in present_weights.items()}

        team_elo = sum(normed[pos] * details[pos] for pos in present_weights)
        rows.append({
            "Verein": team_de,
            "Team-Elo": team_elo,
            "GK_top1": details["GOALKEEPER"],
            "DEF_top4": details["DEFENDER"],
            "MID_top5": details["MIDFIELDER"],
            "FWD_top3": details["FORWARD"],
            "n_matched": len(players),
        })
    df = pd.DataFrame(rows)

    # EN-Spalte für die direkte Verwendung im Simulator anhängen.
    de_to_en = _de_to_en()
    df["team_en"] = df["Verein"].map(de_to_en)
    return df.sort_values("Team-Elo", ascending=False).reset_index(drop=True)


def load_team_elo_dict() -> dict[str, float]:
    """Liest ``team_elo_from_players.csv`` und gibt ``{team_en: rating}`` zurück.

    Wird die Datei nicht gefunden, regenerieren wir sie aus
    ``player_elo_values.csv``.
    """
    if not OUT_PATH.exists():
        if not ELO_PATH.exists():
            raise FileNotFoundError(
                f"{ELO_PATH} fehlt — bitte zuerst `python3 Code/player_elo_values.py` "
                "laufen lassen."
            )
        df = compute_team_elo(pd.read_csv(ELO_PATH, sep=";"))
        write_csv(df)
    else:
        df = pd.read_csv(OUT_PATH, sep=";")
    missing = df[df["team_en"].isna()]
    if len(missing):
        raise KeyError(
            f"Kein EN-Mapping für: {sorted(missing['Verein'])}. "
            "expected_player_games.SIM_TEAM_TO_DE ergänzen."
        )
    return dict(zip(df["team_en"], df["Team-Elo"]))


def write_csv(df: pd.DataFrame, path: Path = OUT_PATH) -> None:
    cols = ["Verein", "team_en", "Team-Elo",
            "GK_top1", "DEF_top4", "MID_top5", "FWD_top3", "n_matched"]
    out = df[cols].copy()
    for c in ("Team-Elo", "GK_top1", "DEF_top4", "MID_top5", "FWD_top3"):
        out[c] = out[c].round(1)
    out.to_csv(path, sep=";", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    if not ELO_PATH.exists():
        raise SystemExit(
            f"{ELO_PATH} fehlt — bitte zuerst `python3 Code/player_elo_values.py`."
        )
    player_elo = pd.read_csv(ELO_PATH, sep=";")
    df = compute_team_elo(player_elo)
    write_csv(df)

    print(f"Teams aggregiert: {len(df)}")
    print(f"Ausgabe:          {OUT_PATH}\n")
    print("Top 15:")
    print(df.head(15)[
        ["Verein", "Team-Elo", "GK_top1", "DEF_top4", "MID_top5", "FWD_top3", "n_matched"]
    ].to_string(index=False))
    print("\nBottom 10:")
    print(df.tail(10)[
        ["Verein", "Team-Elo", "GK_top1", "DEF_top4", "MID_top5", "FWD_top3", "n_matched"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
