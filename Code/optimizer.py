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

BUDGET = 70_000_000
POSITION_QUOTA = {
    "GOALKEEPER": 2,
    "DEFENDER": 5,
    "MIDFIELDER": 5,
    "FORWARD": 3,
}
SQUAD_SIZE = sum(POSITION_QUOTA.values())


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


def attach_expected_games(
    players: pd.DataFrame, path: Path = EXPECTED_GAMES_PATH,
) -> pd.DataFrame:
    """Mergt die Spalte ``Erwartete Spiele`` aus dem Sim+Lineup-Output an.

    Erzeugt sie via ``expected_player_games`` falls die CSV fehlt.
    """
    if not path.exists():
        # Lazy import — vermeidet, dass der Default-Workflow von
        # `expected_player_games` (und damit dem Simulator) abhängt.
        from expected_player_games import build_expected_player_games, write_csv
        write_csv(build_expected_player_games())
    eg = pd.read_csv(path, sep=";")[["ID", "Erwartete Spiele"]]
    out = players.merge(eg, on="ID", how="left")
    out["Erwartete Spiele"] = out["Erwartete Spiele"].fillna(0.0)
    return out


def optimize(
    players: pd.DataFrame,
    objective_col: str = "Punkte",
    minimize: bool = False,
    exclude_zero_objective: bool = False,
    budget: int = BUDGET,
) -> Result:
    df = players.copy()
    if exclude_zero_objective:
        df = df[df[objective_col] > 0].reset_index(drop=True)

    prob = pulp.LpProblem(
        "kicker-manager", pulp.LpMinimize if minimize else pulp.LpMaximize
    )

    x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in df.index]

    prob += pulp.lpSum(df.loc[i, objective_col] * x[i] for i in df.index)

    # Budget
    prob += pulp.lpSum(df.loc[i, "Marktwert"] * x[i] for i in df.index) <= budget

    # Positions-Kontingente (exakt)
    for pos, quota in POSITION_QUOTA.items():
        prob += (
            pulp.lpSum(x[i] for i in df.index if df.loc[i, "Position"] == pos) == quota
        )

    # Squad-Größe (redundant, aber explizit)
    prob += pulp.lpSum(x) == SQUAD_SIZE

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Solver lieferte keinen optimalen Status: {pulp.LpStatus[status]}")

    chosen = [i for i in df.index if x[i].value() > 0.5]
    picks = df.loc[chosen].copy()
    picks = picks.sort_values(
        ["Position", objective_col], ascending=[True, minimize]
    ).reset_index(drop=True)

    return Result(
        objective_value=float(pulp.value(prob.objective)),
        total_cost=int(picks["Marktwert"].sum()),
        picks=picks,
    )


def format_result(result: Result, objective_col: str) -> str:
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
    lines.append(f"Gesamtkosten: {result.total_cost:,} € (Budget: {BUDGET:,} €)")
    lines.append(f"Restbudget:   {BUDGET - result.total_cost:,} €")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_DEFAULT, help="Pfad zur Spieler-CSV")
    parser.add_argument(
        "--objective",
        default="Punkte",
        choices=["Punkte", "Notendurchschnitt", "Erwartete Spiele"],
        help="Zu optimierende Spalte",
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
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args()

    players = load_players(args.data)
    if args.objective == "Erwartete Spiele":
        players = attach_expected_games(players)
    result = optimize(
        players,
        objective_col=args.objective,
        minimize=args.minimize,
        exclude_zero_objective=args.exclude_zero,
        budget=args.budget,
    )
    print(format_result(result, args.objective))


if __name__ == "__main__":
    main()
