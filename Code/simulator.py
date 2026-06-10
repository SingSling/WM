"""Monte-Carlo-Simulator für die WM 2026.

Poisson-Tormodell mit Elo-basierten Erwartungswerten. Die K.o.-Phase
löst dynamisch alle Bracket-Slots (inkl. der "besten 8 Gruppendritten")
und gibt einen vollständigen Turnierverlauf zurück.

Aufruf::

    python3 Code/simulator.py --runs 10000
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from ratings import load_ratings
from schedule import load_schedule, Schedule


# --- Modellparameter ----------------------------------------------------------
#
# Skalenunabhängigkeit: Wir rechnen mit z-Scores (Differenz / Standardabw.)
# der Ratings, nicht mit absoluten Differenzen. So liefern beliebige
# Bewertungsskalen (Elo, 0–100, 1–10, …) identische Vorhersagen — solange
# die Spreizung der Ratings ähnliche Information transportiert.
#
# BETA_PER_SIGMA = 0.30 wurde so gewählt, dass das Verhalten mit
# Default-Elo identisch zur früheren Version (ELO_SCALE = 0.0017) bleibt:
#   ELO_SCALE * σ_Elo = 0.0017 * 176.5 ≈ 0.30

MU_BASE = 1.4               # mittlere Tore pro Team und Spiel (WM-Mittel)
BETA_PER_SIGMA = 0.30       # exp(BETA · z_diff) skaliert Lambda
MAX_GOALS = 12              # Cap, um exotische Poisson-Ausreißer zu bremsen
ET_GOAL_FACTOR = 0.30       # Verlängerung: 30 Minuten ≈ 1/3 der regulären Zeit
SIGMA_FLOOR = 1e-6          # Schutz gegen identische Ratings


@dataclass
class MatchResult:
    team1: str
    team2: str
    goals1: int
    goals2: int
    winner: str | None
    decided_by: str   # "regulation" | "extra_time" | "penalties"


# --- Modell-Primitive ---------------------------------------------------------

def compute_sigma(ratings: dict[str, float]) -> float:
    """Standardabweichung der Ratings — definiert die Skala der z-Scores."""
    vals = np.array(list(ratings.values()), dtype=float)
    sigma = float(vals.std(ddof=1))
    return max(sigma, SIGMA_FLOOR)


def lambdas(rating_a: float, rating_b: float, sigma: float, mu: float = MU_BASE) -> tuple[float, float]:
    z_diff = (rating_a - rating_b) / sigma
    factor = math.exp(BETA_PER_SIGMA * z_diff)
    return mu * factor, mu / factor


def sample_goals(lam_a: float, lam_b: float, rng: np.random.Generator) -> tuple[int, int]:
    g1 = min(int(rng.poisson(lam_a)), MAX_GOALS)
    g2 = min(int(rng.poisson(lam_b)), MAX_GOALS)
    return g1, g2


def simulate_match(
    team1: str, team2: str, ratings: dict[str, float], sigma: float,
    rng: np.random.Generator, knockout: bool = False,
) -> MatchResult:
    la, lb = lambdas(ratings[team1], ratings[team2], sigma)
    g1, g2 = sample_goals(la, lb, rng)

    if not knockout or g1 != g2:
        winner = team1 if g1 > g2 else team2 if g2 > g1 else None
        return MatchResult(team1, team2, g1, g2, winner, "regulation")

    # K.o. + Unentschieden -> Verlängerung
    eg1, eg2 = sample_goals(la * ET_GOAL_FACTOR, lb * ET_GOAL_FACTOR, rng)
    g1, g2 = g1 + eg1, g2 + eg2
    if g1 != g2:
        winner = team1 if g1 > g2 else team2
        return MatchResult(team1, team2, g1, g2, winner, "extra_time")

    # Elfmeterschießen: leicht zugunsten des stärkeren Teams.
    # Skala wie reguläres Modell: z-Score-basiert.
    z = (ratings[team1] - ratings[team2]) / sigma
    p_team1 = 1.0 / (1.0 + math.exp(-z * 0.5))
    winner = team1 if rng.random() < p_team1 else team2
    return MatchResult(team1, team2, g1, g2, winner, "penalties")


# --- Gruppenphase -------------------------------------------------------------

def _empty_standings(teams: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"team": list(teams), "P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
    ).set_index("team")


def _apply_result(table: pd.DataFrame, r: MatchResult) -> None:
    table.at[r.team1, "P"] += 1
    table.at[r.team2, "P"] += 1
    table.at[r.team1, "GF"] += r.goals1
    table.at[r.team1, "GA"] += r.goals2
    table.at[r.team2, "GF"] += r.goals2
    table.at[r.team2, "GA"] += r.goals1
    if r.goals1 > r.goals2:
        table.at[r.team1, "W"] += 1
        table.at[r.team2, "L"] += 1
        table.at[r.team1, "Pts"] += 3
    elif r.goals2 > r.goals1:
        table.at[r.team2, "W"] += 1
        table.at[r.team1, "L"] += 1
        table.at[r.team2, "Pts"] += 3
    else:
        table.at[r.team1, "D"] += 1
        table.at[r.team2, "D"] += 1
        table.at[r.team1, "Pts"] += 1
        table.at[r.team2, "Pts"] += 1


def _rank_table(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    table["GD"] = table["GF"] - table["GA"]
    # Tiebreaker: Pts -> GD -> GF -> Zufall (Platzhalter für H2H/Fair Play)
    table["_rand"] = rng.random(len(table))
    ranked = table.sort_values(
        by=["Pts", "GD", "GF", "_rand"], ascending=[False, False, False, False]
    ).drop(columns="_rand")
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def simulate_group(
    group: str, schedule: Schedule, ratings: dict[str, float], sigma: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    teams = schedule.groups[group]
    table = _empty_standings(teams)
    matches = schedule.group_matches[schedule.group_matches["group"] == group]
    for m in matches.itertuples():
        r = simulate_match(m.team1, m.team2, ratings, sigma, rng, knockout=False)
        _apply_result(table, r)
    ranked = _rank_table(table, rng)
    ranked["group"] = group
    return ranked


# --- Zuordnung der 8 besten Dritten zu den R32-Slots --------------------------

def select_best_thirds_and_assign(
    thirds: pd.DataFrame, schedule: Schedule, rng: np.random.Generator,
) -> dict[str, str]:
    """Wählt die 8 besten Gruppendritten und ordnet sie den 8 R32-Slots zu.

    Slots sind 8 K.o.-Spiele mit Slot-Typ ``third_place`` und Constraint
    ``from_groups``. Wir lösen ein bipartites Matching, sodass jeder Slot
    einen Drittplatzierten erhält, dessen Gruppe in ``from_groups`` enthalten ist.
    Bevorzugt wird der jeweils höchstplatzierte Drittplatzierte.
    """
    thirds = thirds.copy()
    thirds["_rand"] = rng.random(len(thirds))
    thirds_sorted = thirds.sort_values(
        by=["Pts", "GD", "GF", "_rand"], ascending=[False, False, False, False]
    ).drop(columns="_rand").head(8)

    slot_matches = [
        m for m in schedule.raw["matches"]
        if any(ref.get("type") == "third_place"
               for ref in (m.get("team1_ref"), m.get("team2_ref")) if isinstance(ref, dict))
    ]
    # (match_no, from_groups)
    slots: list[tuple[int, list[str]]] = []
    for m in slot_matches:
        for key in ("team1_ref", "team2_ref"):
            ref = m.get(key)
            if isinstance(ref, dict) and ref.get("type") == "third_place":
                slots.append((m["match_no"], ref["from_groups"]))

    assert len(slots) == 8, f"Erwarte 8 third-place Slots, fand {len(slots)}"

    n = 8
    # Kostenmatrix: niedrige Kosten = bevorzugte Zuordnung.
    # Cost = -rank_score, so dass top-thirds bevorzugt werden. Unzulässige
    # Kombinationen erhalten sehr hohe Kosten.
    cost = np.full((n, n), 1e9)
    third_records = list(thirds_sorted.iterrows())  # [(team, row), ...]
    for i, (team, row) in enumerate(third_records):
        for j, (_match_no, allowed_groups) in enumerate(slots):
            if row["group"] in allowed_groups:
                cost[i, j] = -(row["Pts"] * 100 + row["GD"] * 10 + row["GF"])

    row_ind, col_ind = linear_sum_assignment(cost)
    if cost[row_ind, col_ind].max() > 1e8:
        # Keine zulässige Zuordnung — sollte praktisch nie passieren
        raise RuntimeError("Keine gültige Zuordnung der Gruppendritten möglich")

    assignment: dict[str, str] = {}  # key "M{match_no}_third" -> team
    for i, j in zip(row_ind, col_ind):
        team = third_records[i][0]
        match_no = slots[j][0]
        assignment[f"M{match_no}_third"] = team
    return assignment


# --- Vollständige Turnier-Simulation ------------------------------------------

def _resolve_ref(
    ref: dict, group_results: dict[str, pd.DataFrame],
    match_results: dict[int, MatchResult], third_assignment: dict[str, str], match_no: int,
) -> str:
    t = ref["type"]
    if t == "winner":
        return group_results[ref["group"]].iloc[0].name
    if t == "runner_up":
        return group_results[ref["group"]].iloc[1].name
    if t == "third_place":
        return third_assignment[f"M{match_no}_third"]
    if t == "match_winner":
        return match_results[ref["match_no"]].winner  # type: ignore[return-value]
    if t == "match_loser":
        mr = match_results[ref["match_no"]]
        return mr.team1 if mr.winner == mr.team2 else mr.team2
    raise ValueError(f"Unbekannter Ref-Typ: {t}")


def simulate_tournament(
    schedule: Schedule, ratings: dict[str, float], rng: np.random.Generator,
    sigma: float | None = None,
) -> dict:
    if sigma is None:
        sigma = compute_sigma(ratings)

    # Gruppenphase
    group_tables: dict[str, pd.DataFrame] = {}
    for g in schedule.groups:
        group_tables[g] = simulate_group(g, schedule, ratings, sigma, rng)

    # Beste Dritte → R32-Slots
    thirds = pd.concat([t.iloc[[2]] for t in group_tables.values()])
    third_assignment = select_best_thirds_and_assign(thirds, schedule, rng)

    # K.o.-Phase
    match_results: dict[int, MatchResult] = {}
    ko_sorted = sorted(
        (m for m in schedule.raw["matches"] if m["stage"] != "group"),
        key=lambda m: m["match_no"],
    )
    for m in ko_sorted:
        t1 = _resolve_ref(m["team1_ref"], group_tables, match_results, third_assignment, m["match_no"])
        t2 = _resolve_ref(m["team2_ref"], group_tables, match_results, third_assignment, m["match_no"])
        match_results[m["match_no"]] = simulate_match(t1, t2, ratings, sigma, rng, knockout=True)

    champion = match_results[104].winner
    runner_up = (match_results[104].team1
                 if match_results[104].winner == match_results[104].team2
                 else match_results[104].team2)
    third = match_results[103].winner

    return {
        "groups": group_tables,
        "third_assignment": third_assignment,
        "matches": match_results,
        "champion": champion,
        "runner_up": runner_up,
        "third": third,
    }


# --- Monte-Carlo --------------------------------------------------------------

def run_monte_carlo(
    schedule: Schedule, ratings: dict[str, float], n_runs: int = 10_000, seed: int = 42,
    progress_callback=None,
) -> pd.DataFrame:
    """Monte-Carlo-Aggregation über ``n_runs`` Turniere.

    ``progress_callback(i, n_runs)`` wird – falls gesetzt – nach jedem
    Iterationsschritt aufgerufen (``i`` ist 1-indiziert, ``i == n_runs`` am
    Ende). Nützlich z.B. für ``st.progress`` im Dashboard.
    """
    rng = np.random.default_rng(seed)
    sigma = compute_sigma(ratings)
    counts = defaultdict(lambda: Counter())  # stage -> Counter(team -> reached)
    qualified_counter: Counter[str] = Counter()  # erreichte R32 (= K.o.-Quali)
    champion_counter: Counter[str] = Counter()
    runner_up_counter: Counter[str] = Counter()
    podium_counter: Counter[str] = Counter()
    # Summe (über alle Sims) der gesamten Tor-Differenz pro Team — Gruppe und
    # K.o. zusammen. Penalty-Shootouts zählen nicht (Goals1/Goals2 enthalten
    # nur Tore aus regulärer Spielzeit + Verlängerung).
    gd_sum: defaultdict[str, float] = defaultdict(float)

    stage_of_match = {m["match_no"]: m["stage"] for m in schedule.raw["matches"]}

    for i in range(n_runs):
        outcome = simulate_tournament(schedule, ratings, rng, sigma=sigma)

        # Sieger pro Stage zählen → bestimmt p_r16/p_qf/p_sf/p_final
        for match_no, mr in outcome["matches"].items():
            survivor = mr.winner
            if survivor:
                counts[stage_of_match[match_no]][survivor] += 1

        # K.o.-Teilnehmer: jedes Team, das in einem R32-Spiel steht
        for match_no, mr in outcome["matches"].items():
            if stage_of_match[match_no] == "R32":
                qualified_counter[mr.team1] += 1
                qualified_counter[mr.team2] += 1

        champion_counter[outcome["champion"]] += 1
        runner_up_counter[outcome["runner_up"]] += 1
        podium_counter[outcome["champion"]] += 1
        podium_counter[outcome["runner_up"]] += 1
        podium_counter[outcome["third"]] += 1

        # Tor-Differenz aufsummieren: Gruppen-Tabellen tragen GD bereits,
        # K.o.-Spiele addieren wir aus den MatchResults.
        for table in outcome["groups"].values():
            for team_name, gd in table["GD"].items():
                gd_sum[team_name] += float(gd)
        for mr in outcome["matches"].values():
            gd_sum[mr.team1] += mr.goals1 - mr.goals2
            gd_sum[mr.team2] += mr.goals2 - mr.goals1

        if progress_callback is not None:
            progress_callback(i + 1, n_runs)

    # Aggregierte Wahrscheinlichkeiten pro Team
    rows = []
    teams = sorted({t for ts in schedule.groups.values() for t in ts})
    for team in teams:
        p_qualified = qualified_counter.get(team, 0) / n_runs
        p_r16   = counts["R32"].get(team, 0) / n_runs
        p_qf    = counts["R16"].get(team, 0) / n_runs
        p_sf    = counts["QF"].get(team, 0) / n_runs
        p_final = counts["SF"].get(team, 0) / n_runs
        # Erwartete Spiele:
        #   3 Gruppenspiele + 1 R32 (falls qualifiziert) + 1 R16 + 1 QF + 1 SF
        #   + 1 Final ODER 3rd-Place (beide SF-Teams spielen ein weiteres Spiel)
        exp_games = 3 + p_qualified + p_r16 + p_qf + 2 * p_sf
        rows.append({
            "team": team,
            "rating": ratings[team],
            "p_winner":  champion_counter.get(team, 0) / n_runs,
            "exp_games": exp_games,
            "exp_gd": gd_sum.get(team, 0.0) / n_runs,
            "p_qualified": p_qualified,
            "p_r16":   p_r16,
            "p_qf":    p_qf,
            "p_sf":    p_sf,
            "p_final": p_final,
            "p_podium": podium_counter.get(team, 0) / n_runs,
        })
    return pd.DataFrame(rows).sort_values("p_winner", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    schedule = load_schedule()
    ratings = load_ratings()

    df = run_monte_carlo(schedule, ratings, n_runs=args.runs, seed=args.seed)
    print(f"\nMonte-Carlo: {args.runs:,} Simulationen\n")

    headline = df[["team", "rating", "p_winner", "exp_games", "exp_gd"]].copy()
    headline["p_winner"] = (headline["p_winner"] * 100).round(2).astype(str) + "%"
    headline["exp_games"] = headline["exp_games"].round(2)
    headline["exp_gd"] = headline["exp_gd"].round(2)
    headline = headline.rename(columns={
        "rating": "Rating", "p_winner": "Titel-Wkt.",
        "exp_games": "Ø Spiele", "exp_gd": "Ø TD",
    })

    print("Top 15:")
    print(headline.head(15).to_string(index=False))
    print(f"\nBottom 10 (von {len(df)}):")
    print(headline.tail(10).to_string(index=False))

    # Sanity-Checks
    total_exp = df["exp_games"].sum()
    # 72 Gruppenspiele (2 Teams je) + 32 K.o. (2 Teams je) = 208 Team-Spiele
    print(f"\nΣ erwartete Spiele über alle Teams: {total_exp:.1f}  (exakt: 208)")
    # Tor-Differenz ist ein Nullsummenspiel → Summe ≈ 0
    print(f"Σ erwartete Tor-Differenz: {df['exp_gd'].sum():+.2f}  (erwartet: 0)")


if __name__ == "__main__":
    main()
