"""Kicker Manager Squad Optimizer.

Wählt aus einer Spielerliste das punktemaximale Team unter Berücksichtigung
von Budget- und Positions-Nebenbedingungen via Integer Linear Programming.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pulp


DATA_DEFAULT = Path(__file__).resolve().parent.parent / "Data" / "players-se-k01012026.csv"
EXPECTED_GAMES_PATH = Path(__file__).resolve().parent.parent / "Data" / "expected_player_games.csv"
TM_VALUES_PATH = Path(__file__).resolve().parent.parent / "Data" / "transfermarkt_values.csv"

# Volle Kicker-Manager-Squad und Budget: 15 Spieler / 70 Mio €. Der Optimizer
# wählt aber nur die Startelf (11 Spieler) — die 4 Bank-Plätze füllt der User
# manuell mit 1-Mio-Spielern. Das reserviert 4 × 1 Mio = 4 Mio von der Squad-
# Kasse für die Bank; der Solver bekommt 66 Mio.
SQUAD_BUDGET = 70_000_000
BENCH_FILLER_COST = 1_000_000
BENCH_SIZE = 4
STARTING_XI_BUDGET = SQUAD_BUDGET - BENCH_SIZE * BENCH_FILLER_COST  # 66_000_000

# Erlaubte Formationen (immer 1 GK). Schlüssel ist die übliche Schreibweise
# DEF-MID-FWD, der Wert die exakte Positions-Quote für den Solver.
FORMATIONS: dict[str, dict[str, int]] = {
    "4-4-2": {"GOALKEEPER": 1, "DEFENDER": 4, "MIDFIELDER": 4, "FORWARD": 2},
    "3-4-3": {"GOALKEEPER": 1, "DEFENDER": 3, "MIDFIELDER": 4, "FORWARD": 3},
    "3-5-2": {"GOALKEEPER": 1, "DEFENDER": 3, "MIDFIELDER": 5, "FORWARD": 2},
    "4-3-3": {"GOALKEEPER": 1, "DEFENDER": 4, "MIDFIELDER": 3, "FORWARD": 3},
    "4-5-1": {"GOALKEEPER": 1, "DEFENDER": 4, "MIDFIELDER": 5, "FORWARD": 1},
    "5-3-2": {"GOALKEEPER": 1, "DEFENDER": 5, "MIDFIELDER": 3, "FORWARD": 2},
    "5-4-1": {"GOALKEEPER": 1, "DEFENDER": 5, "MIDFIELDER": 4, "FORWARD": 1},
}
DEFAULT_FORMATION = "3-4-3"

# Default-Quote für den Solver (entspricht DEFAULT_FORMATION). Aus
# Rückwärtskompatibilitäts-/Convenience-Gründen weiterhin als POSITION_QUOTA
# exportiert — das Dashboard kann via ``formation``-Argument abweichen.
POSITION_QUOTA = FORMATIONS[DEFAULT_FORMATION]
SQUAD_SIZE = sum(POSITION_QUOTA.values())  # 11
# Alias für Bestandsskripte/Dashboard, die ``BUDGET`` als Solver-Budget nutzen.
BUDGET = STARTING_XI_BUDGET


@dataclass
class Result:
    objective_value: float
    total_cost: int
    picks: pd.DataFrame


def load_players(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    # Filter offensichtliche Datenfehler (z.B. Marktwert 999.000.000).
    df = df[df["Marktwert"] < 100_000_000].reset_index(drop=True)
    # Sicherstellen, dass alle Pflicht-Positionen vertreten sind.
    missing = set(POSITION_QUOTA) - set(df["Position"].unique())
    if missing:
        raise ValueError(f"Positionen fehlen im Datensatz: {missing}")
    return df


EXPECTED_PLAYER_COLS = ("Erwartete Spiele", "Erwartete Tordifferenz")


def attach_expected_metrics(
    players: pd.DataFrame,
    path: Path = EXPECTED_GAMES_PATH,
    columns: tuple[str, ...] = EXPECTED_PLAYER_COLS,
) -> pd.DataFrame:
    """Mergt vorberechnete Erwartungswert-Spalten aus dem Sim+Lineup-Output an.

    Erzeugt die CSV via ``expected_player_games`` falls sie fehlt.
    """
    if not path.exists():
        # Lazy import — vermeidet, dass der Default-Workflow von
        # `expected_player_games` (und damit dem Simulator) abhängt.
        from expected_player_games import build_expected_player_metrics, write_csv
        write_csv(build_expected_player_metrics())
    eg = pd.read_csv(path, sep=";")
    keep = ["ID"] + [c for c in columns if c in eg.columns]
    out = players.merge(eg[keep], on="ID", how="left")
    for c in columns:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out


# Back-compat-Alias.
attach_expected_games = attach_expected_metrics


def attach_tm_values(
    players: pd.DataFrame, path: Path = TM_VALUES_PATH,
) -> pd.DataFrame:
    """Mergt die Spalte ``Marktwert TM`` aus dem TM-Join-CSV an.

    Spieler ohne TM-Match erhalten ``Marktwert TM = 0`` — der Optimizer
    wird sie damit beim Maximieren von Marktwert TM nicht aktiv wählen
    (es sei denn, sie sind gepinnt oder die Formations-Quote zwingt sie).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} fehlt — bitte zuerst `python3 Code/transfermarkt_values.py` "
            "laufen lassen."
        )
    tm = pd.read_csv(path, sep=";")[["ID", "Marktwert TM"]]
    out = players.merge(tm, on="ID", how="left")
    out["Marktwert TM"] = out["Marktwert TM"].fillna(0.0)
    return out


def optimize(
    players: pd.DataFrame,
    objective_col: str = "Punkte",
    minimize: bool = False,
    exclude_zero_objective: bool = False,
    budget: int = BUDGET,
    position_quota: dict[str, int] | None = None,
    lock_ids: list[str] | tuple[str, ...] | None = None,
) -> Result:
    """Wählt eine punktemaximale Startelf unter Budget- und Quoten-Constraints.

    ``lock_ids`` zwingt den Solver, die genannten Spieler zu wählen
    (x_i = 1). Ihre Marktwerte fließen normal ins Budget ein, ihre
    Positionen normal in die Quote. Locked-Spieler überleben den
    ``exclude_zero_objective``-Filter, damit reservistenartige Pins nicht
    versehentlich rausfallen.
    """
    if position_quota is None:
        position_quota = POSITION_QUOTA
    squad_size = sum(position_quota.values())
    lock_set: set[str] = set(lock_ids or [])

    df = players.copy()
    if exclude_zero_objective:
        keep = df[objective_col] > 0
        if lock_set:
            keep = keep | df["ID"].isin(lock_set)
        df = df[keep].reset_index(drop=True)

    if lock_set:
        missing_locks = lock_set - set(df["ID"])
        if missing_locks:
            raise ValueError(
                f"Gepinnte Spieler nicht im Datensatz: {sorted(missing_locks)}"
            )

    prob = pulp.LpProblem(
        "kicker-manager", pulp.LpMinimize if minimize else pulp.LpMaximize
    )

    x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in df.index]

    prob += pulp.lpSum(df.loc[i, objective_col] * x[i] for i in df.index)

    # Budget — inkl. gepinnter Spieler.
    prob += pulp.lpSum(df.loc[i, "Marktwert"] * x[i] for i in df.index) <= budget

    # Positions-Kontingente (exakt) — inkl. gepinnter Spieler.
    for pos, quota in position_quota.items():
        prob += (
            pulp.lpSum(x[i] for i in df.index if df.loc[i, "Position"] == pos) == quota
        )

    # Squad-Größe (redundant, aber explizit)
    prob += pulp.lpSum(x) == squad_size

    # Lock-Constraints.
    if lock_set:
        id_to_idx = {df.loc[i, "ID"]: i for i in df.index}
        for pid in lock_set:
            prob += x[id_to_idx[pid]] == 1

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Solver lieferte keinen optimalen Status: {pulp.LpStatus[status]}")

    chosen = [i for i in df.index if x[i].value() > 0.5]
    picks = df.loc[chosen].copy()
    picks["Gepinnt"] = picks["ID"].isin(lock_set)
    picks = picks.sort_values(
        ["Position", objective_col], ascending=[True, minimize]
    ).reset_index(drop=True)

    return Result(
        objective_value=float(pulp.value(prob.objective)),
        total_cost=int(picks["Marktwert"].sum()),
        picks=picks,
    )


def format_result(
    result: Result, objective_col: str, budget: int = BUDGET,
) -> str:
    lines = []
    cols = ["Position", "Angezeigter Name", "Verein", "Marktwert", "Punkte", "Notendurchschnitt"]
    if objective_col not in cols:
        cols.append(objective_col)
    for pos in ["GOALKEEPER", "DEFENDER", "MIDFIELDER", "FORWARD"]:
        block = result.picks[result.picks["Position"] == pos][cols]
        lines.append(f"\n=== {pos} ({len(block)}) ===")
        lines.append(block.to_string(index=False))
    lines.append("")
    lines.append(f"Zielwert ({objective_col}): {result.objective_value:.2f}")
    lines.append(f"Gesamtkosten: {result.total_cost:,} € (Budget: {budget:,} €)")
    lines.append(f"Restbudget:   {budget - result.total_cost:,} €")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT, help="Pfad zur Spieler-CSV")
    parser.add_argument(
        "--objective",
        default="Punkte",
        choices=[
            "Punkte", "Notendurchschnitt",
            "Erwartete Spiele", "Erwartete Tordifferenz",
            "Marktwert TM",
        ],
        help="Zu optimierende Spalte",
    )
    parser.add_argument(
        "--formation",
        default=DEFAULT_FORMATION,
        choices=list(FORMATIONS),
        help="Startformation (1 GK + DEF-MID-FWD)",
    )
    parser.add_argument(
        "--minimize",
        action="store_true",
        help="Zielwert minimieren statt maximieren (z.B. Notendurchschnitt)",
    )
    parser.add_argument(
        "--exclude-zero",
        action="store_true",
        help="Spieler mit Zielwert == 0 ausschließen (sinnvoll bei Notendurchschnitt)",
    )
    parser.add_argument(
        "--budget", type=int, default=STARTING_XI_BUDGET,
        help=f"Solver-Budget für die Startelf (Default {STARTING_XI_BUDGET:,} € = "
             f"{SQUAD_BUDGET:,} € Squad-Kasse − {BENCH_SIZE} × {BENCH_FILLER_COST:,} € Bank)",
    )
    args = parser.parse_args()

    players = load_players(args.data)
    if args.objective in EXPECTED_PLAYER_COLS:
        players = attach_expected_metrics(players)
    elif args.objective == "Marktwert TM":
        players = attach_tm_values(players)
    result = optimize(
        players,
        objective_col=args.objective,
        minimize=args.minimize,
        exclude_zero_objective=args.exclude_zero,
        budget=args.budget,
        position_quota=FORMATIONS[args.formation],
    )
    print(f"Formation: {args.formation}")
    print(format_result(result, args.objective, budget=args.budget))


if __name__ == "__main__":
    main()
