"""Loader und Helfer für den WM 2026 Spielplan.

Stellt die in ``Data/wm_schedule.json`` gespeicherten Daten als Pandas
Dataframes zur Verfügung. Wird die Grundlage für die spätere
Turnier-Simulation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SCHEDULE_PATH = Path(__file__).resolve().parent.parent / "Data" / "wm_schedule.json"


@dataclass(frozen=True)
class Schedule:
    raw: dict
    groups: dict[str, list[str]]
    group_matches: pd.DataFrame   # Spalten: match_no, group, date, time_uk, venue, team1, team2
    knockout_matches: pd.DataFrame  # Spalten: match_no, stage, date, time_uk, venue, team1_ref, team2_ref
    teams: list[str]


def load_schedule(path: Path = SCHEDULE_PATH) -> Schedule:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    groups = raw["groups"]

    rows_group, rows_ko = [], []
    for m in raw["matches"]:
        if m["stage"] == "group":
            rows_group.append(m)
        else:
            rows_ko.append(m)

    group_df = pd.DataFrame(rows_group)
    ko_df = pd.DataFrame(rows_ko)

    teams = sorted({t for ts in groups.values() for t in ts})
    return Schedule(raw=raw, groups=groups, group_matches=group_df, knockout_matches=ko_df, teams=teams)


def describe_ref(ref: dict) -> str:
    """Mensch-lesbare Repräsentation eines Bracket-Slots."""
    t = ref["type"]
    if t == "winner":
        return f"Sieger Gruppe {ref['group']}"
    if t == "runner_up":
        return f"Zweiter Gruppe {ref['group']}"
    if t == "third_place":
        return f"3. aus {'/'.join(ref['from_groups'])}"
    if t == "match_winner":
        return f"Sieger Spiel {ref['match_no']}"
    if t == "match_loser":
        return f"Verlierer Spiel {ref['match_no']}"
    raise ValueError(f"Unbekannter Ref-Typ: {t}")


def validate(schedule: Schedule) -> list[str]:
    """Konsistenzprüfungen — gibt eine Liste von Warnungen zurück (leer = ok)."""
    warnings: list[str] = []

    # 1) Gruppen-Größen
    for g, ts in schedule.groups.items():
        if len(ts) != 4:
            warnings.append(f"Gruppe {g} hat {len(ts)} Teams (erwartet: 4)")

    # 2) 6 Gruppenspiele pro Gruppe
    counts = schedule.group_matches.groupby("group").size()
    for g, n in counts.items():
        if n != 6:
            warnings.append(f"Gruppe {g} hat {n} Spiele (erwartet: 6)")

    # 3) Jede Gruppen-Paarung kommt genau einmal vor (4 über 2 = 6 Paare)
    for g, ts in schedule.groups.items():
        sub = schedule.group_matches[schedule.group_matches["group"] == g]
        pairs = {frozenset((r.team1, r.team2)) for r in sub.itertuples()}
        expected = {frozenset((a, b)) for i, a in enumerate(ts) for b in ts[i + 1 :]}
        missing = expected - pairs
        extra = pairs - expected
        for p in missing:
            warnings.append(f"Gruppe {g}: fehlende Paarung {tuple(p)}")
        for p in extra:
            warnings.append(f"Gruppe {g}: unerwartete Paarung {tuple(p)}")

    # 4) Spielnummern lückenlos 1..104
    all_nos = sorted(m["match_no"] for m in schedule.raw["matches"])
    if all_nos != list(range(1, 105)):
        warnings.append(f"Spielnummern unvollständig: {set(range(1, 105)) - set(all_nos)}")

    return warnings


if __name__ == "__main__":
    s = load_schedule()
    print(f"Geladen: {len(s.teams)} Teams, {len(s.group_matches)} Gruppenspiele, "
          f"{len(s.knockout_matches)} K.o.-Spiele")
    issues = validate(s)
    if issues:
        print("\nWarnungen:")
        for w in issues:
            print(" -", w)
    else:
        print("\nKonsistenz-Check: OK")

    print("\nErste 5 Gruppenspiele:")
    print(s.group_matches.head().to_string(index=False))

    print("\nFinale (Spiel 104):")
    final = s.knockout_matches[s.knockout_matches["match_no"] == 104].iloc[0]
    print(f"  {final['date']} {final['time_uk']} UK in {final['venue']}")
    print(f"  {describe_ref(final['team1_ref'])} vs {describe_ref(final['team2_ref'])}")
