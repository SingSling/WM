"""Streamlit Dashboard für den Kicker Manager Optimizer + WM-Simulation."""

from __future__ import annotations

import sys
from pathlib import Path

# Code/ explizit auf sys.path legen — Streamlit Cloud (Python ≥3.14) initialisiert
# das Skript-Verzeichnis nicht zuverlässig vor unseren Top-Level-Imports.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd
import streamlit as st

from expected_player_games import (
    OUT_PATH as EXPECTED_GAMES_PATH,
    apply_expected_metric,
    load_default_probabilities,
    mc_to_team_metric_de,
)
from optimizer import (
    BUDGET,
    DATA_DEFAULT,
    DEFAULT_FORMATION,
    FORMATIONS,
    load_players,
    optimize,
)
from ratings import load_ratings
from schedule import load_schedule
from simulator import run_monte_carlo


POSITION_LABEL = {
    "GOALKEEPER": "Torhüter",
    "DEFENDER":   "Verteidiger",
    "MIDFIELDER": "Mittelfeld",
    "FORWARD":    "Sturm",
}

DISPLAY_COLS = [
    "Angezeigter Name",
    "Verein",
    "Marktwert",
    "Punkte",
    "Notendurchschnitt",
]

OBJ_ERWARTET_DEFAULT = "Erwartete Spiele"
OBJ_ERWARTET_CUSTOM = "Erwartete Spiele (Custom)"
OBJ_TD_DEFAULT = "Erwartete Tordifferenz"
OBJ_TD_CUSTOM = "Erwartete Tordifferenz (Custom)"

# Mapping zwischen dem Default/Custom-Label und (player_col, team_metric_col).
EXPECTED_OBJ_SPECS: dict[str, dict[str, str]] = {
    OBJ_ERWARTET_DEFAULT: {"player_col": "Erwartete Spiele",       "team_col": "exp_games"},
    OBJ_ERWARTET_CUSTOM:  {"player_col": "Erwartete Spiele",       "team_col": "exp_games"},
    OBJ_TD_DEFAULT:       {"player_col": "Erwartete Tordifferenz", "team_col": "exp_gd"},
    OBJ_TD_CUSTOM:        {"player_col": "Erwartete Tordifferenz", "team_col": "exp_gd"},
}
DEFAULT_OBJECTIVES = (OBJ_ERWARTET_DEFAULT, OBJ_TD_DEFAULT)
CUSTOM_OBJECTIVES = (OBJ_ERWARTET_CUSTOM, OBJ_TD_CUSTOM)
EXPECTED_OBJECTIVES = DEFAULT_OBJECTIVES + CUSTOM_OBJECTIVES


# ---------- Caching ----------

@st.cache_data
def cached_load_players(path: str) -> pd.DataFrame:
    return load_players(Path(path))


@st.cache_data
def cached_load_schedule():
    return load_schedule()


@st.cache_data
def cached_default_ratings(source: str = "elo") -> dict[str, float]:
    return load_ratings(source=source)


@st.cache_data
def cached_default_probabilities() -> dict[str, float]:
    return load_default_probabilities()


@st.cache_data
def cached_default_expected_metrics_csv() -> pd.DataFrame:
    """Vorberechnete Erwartungswerte (ID + alle bekannten Spielerspalten)."""
    df = pd.read_csv(EXPECTED_GAMES_PATH, sep=";")
    keep = ["ID"] + [c for c in (
        "Erwartete Spiele", "Erwartete Tordifferenz"
    ) if c in df.columns]
    return df[keep]


RATING_SOURCES = {
    "Elo (World Football)": "elo",
    "WM-Sieger-Quoten (Buchmacher)": "betting",
}


# ---------- Helpers ----------

def format_eur(value: int) -> str:
    return f"{value:,} €".replace(",", ".")


def _attach_default_expected(
    players: pd.DataFrame, player_col: str,
) -> pd.DataFrame:
    """Merge die vorberechnete Erwartungswert-Spalte aus der CSV."""
    eg = cached_default_expected_metrics_csv()
    if player_col not in eg.columns:
        raise KeyError(
            f"Spalte '{player_col}' fehlt in {EXPECTED_GAMES_PATH.name}. "
            "Regeneriere die Datei via expected_player_games.py."
        )
    out = players.merge(eg[["ID", player_col]], on="ID", how="left")
    out[player_col] = out[player_col].fillna(0.0)
    return out


def _attach_custom_expected(
    players: pd.DataFrame, player_col: str, team_col: str,
) -> pd.DataFrame:
    """Verknüpft eigene Sim + eigene Wahrscheinlichkeiten."""
    state_key = f"custom_team_{team_col}"
    if state_key not in st.session_state:
        raise RuntimeError(
            f"Keine Custom-Sim für {team_col} vorhanden. "
            "Erst auf dem Simulator-Tab Simulation starten."
        )
    team_metric: dict[str, float] = st.session_state[state_key]
    probs = dict(cached_default_probabilities())
    probs.update(st.session_state.get("custom_probabilities", {}))
    return apply_expected_metric(players, team_metric, probs, out_col=player_col)


def _custom_eg_available() -> bool:
    # exp_games und exp_gd werden im selben Sim-Lauf gesetzt — eines reicht
    # als Indikator für „Sim ist gelaufen“.
    return "custom_team_exp_games" in st.session_state


# ===================== OPTIMIZER TAB =====================

def render_optimizer() -> None:
    st.subheader("Kader-Optimierung")

    # Verfügbare Ziele zusammenbauen — Custom nur, wenn eine eigene Sim
    # gelaufen ist (Probabilities-Overrides sind optional und fallen zurück).
    objective_options = ["Punkte", "Notendurchschnitt"]
    if EXPECTED_GAMES_PATH.exists():
        objective_options.extend(DEFAULT_OBJECTIVES)
    if _custom_eg_available():
        objective_options.extend(CUSTOM_OBJECTIVES)

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 2, 1.5, 2])
        with c1:
            data_path = st.text_input("CSV-Pfad", value=str(DATA_DEFAULT), key="opt_csv")
        with c2:
            budget = st.number_input(
                "Budget (€)", min_value=10_000_000, max_value=200_000_000,
                value=BUDGET, step=1_000_000, key="opt_budget",
                help=(
                    "Default: 66 Mio (= 70 Mio Squad-Kasse − 4 × 1 Mio "
                    "Bank-Spieler). Der Solver wählt nur die Startelf; "
                    "Bank füllst du manuell."
                ),
            )
        with c3:
            formation = st.selectbox(
                "Formation",
                options=list(FORMATIONS),
                index=list(FORMATIONS).index(DEFAULT_FORMATION),
                key="opt_formation",
            )
        with c4:
            objective = st.selectbox(
                "Zielgröße", options=objective_options, index=0, key="opt_obj",
            )

        # Spieler-Pins: lädt die Spielerliste bewusst INNERHALB der Container-
        # Box, damit die Pin-Auswahl optisch zur Optimierer-Einstellung gehört.
        try:
            _players_for_pin = cached_load_players(data_path)
        except Exception:
            _players_for_pin = None

        if _players_for_pin is not None:
            pin_labels: dict[str, str] = {
                row["ID"]: (
                    f"{row['Angezeigter Name']} — {row['Verein']} "
                    f"({row['Position'][:3]}, {format_eur(int(row['Marktwert']))})"
                )
                for _, row in _players_for_pin.iterrows()
            }
            lock_ids = st.multiselect(
                "Gepinnte Spieler (immer im Kader)",
                options=_players_for_pin["ID"].tolist(),
                format_func=lambda pid: pin_labels.get(pid, pid),
                default=st.session_state.get("opt_locks_default", []),
                placeholder="Spielername zum Filtern tippen…",
                key="opt_locks",
                help=(
                    "Diese Spieler werden in jeder Optimierung gesetzt. Ihre "
                    "Marktwerte fließen ins Budget, ihre Positionen in die "
                    "Formation. Optimizer füllt nur die übrigen Plätze."
                ),
            )
        else:
            lock_ids = []

        # Pin-Validierung: Kosten vs. Budget und Positionen vs. Formation-Quote.
        pin_error = False
        if lock_ids and _players_for_pin is not None:
            locked_df = _players_for_pin[_players_for_pin["ID"].isin(lock_ids)]
            locked_cost = int(locked_df["Marktwert"].sum())
            pos_counts = locked_df["Position"].value_counts().to_dict()
            quota = FORMATIONS[formation]
            over_pos = [
                f"{POSITION_LABEL[p]} ({pos_counts[p]}/{quota.get(p, 0)})"
                for p in pos_counts if pos_counts[p] > quota.get(p, 0)
            ]
            remaining = int(budget) - locked_cost
            st.caption(
                f"Gepinnt: {len(lock_ids)} Spieler · "
                f"Kosten {format_eur(locked_cost)} · "
                f"Restbudget für Solver: {format_eur(remaining)}"
            )
            if locked_cost > budget:
                pin_error = True
                st.error(
                    f"Pin-Kosten {format_eur(locked_cost)} überschreiten das "
                    f"Budget {format_eur(int(budget))}."
                )
            if over_pos:
                pin_error = True
                st.error(
                    "Zu viele gepinnte Spieler auf Positionen: " + ", ".join(over_pos)
                )

        c_opts, c_btn = st.columns([3, 1])
        with c_opts:
            if objective == "Punkte":
                minimize = False
                exclude_zero = st.checkbox(
                    "Spieler mit 0 Punkten ausschließen", value=False, key="opt_zero_p",
                )
            elif objective == "Notendurchschnitt":
                direction = st.radio(
                    "Optimierungsrichtung",
                    options=["Minimieren (beste Note)", "Maximieren"],
                    index=0, horizontal=True, key="opt_dir",
                )
                minimize = direction.startswith("Minimieren")
                exclude_zero = st.checkbox(
                    "Spieler mit Note 0.0 ausschließen (kein Spiel)",
                    value=True, key="opt_zero_n",
                )
            else:
                # Erwartungswert-Ziele (Spiele / Tordifferenz, Default oder
                # Custom): maximieren, 0-Spieler nicht ausschließen (sonst
                # fallen alle Reservisten raus).
                minimize = False
                exclude_zero = False
                if objective in CUSTOM_OBJECTIVES:
                    st.caption(
                        "Custom: Team-Erwartungswerte aus deiner zuletzt "
                        "gelaufenen Simulation + ggf. angepasste Startelf-"
                        "Wahrscheinlichkeiten."
                    )
                else:
                    st.caption(
                        "Default: vorberechnete Werte aus "
                        f"`{EXPECTED_GAMES_PATH.name}` (Elo-Sim × Lineup-"
                        "Wahrscheinlichkeiten)."
                    )
        with c_btn:
            run = st.button("Optimieren", type="primary", use_container_width=True)

    try:
        players = cached_load_players(data_path)
    except Exception as exc:
        st.error(f"CSV konnte nicht geladen werden: {exc}")
        return
    st.caption(f"{len(players)} Spieler geladen aus `{data_path}`")

    if objective in EXPECTED_OBJECTIVES:
        spec = EXPECTED_OBJ_SPECS[objective]
        if objective in DEFAULT_OBJECTIVES:
            players = _attach_default_expected(players, spec["player_col"])
        else:
            players = _attach_custom_expected(
                players, spec["player_col"], spec["team_col"],
            )

    if not run:
        st.info("Einstellungen wählen und „Optimieren“ klicken.")
        return

    if pin_error:
        # Solver würde sicher infeasible sein — Fehler ist oben schon sichtbar.
        return

    objective_col = (
        EXPECTED_OBJ_SPECS[objective]["player_col"]
        if objective in EXPECTED_OBJECTIVES
        else objective
    )

    position_quota = FORMATIONS[formation]

    with st.spinner("Solver läuft…"):
        try:
            result = optimize(
                players, objective_col=objective_col, minimize=minimize,
                exclude_zero_objective=exclude_zero, budget=int(budget),
                position_quota=position_quota,
                lock_ids=lock_ids or None,
            )
        except Exception as exc:
            st.error(f"Optimierung fehlgeschlagen: {exc}")
            return

    picks = result.picks
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Σ {objective}", f"{result.objective_value:.2f}")
    col2.metric("Gesamtkosten", format_eur(result.total_cost))
    col3.metric("Restbudget", format_eur(int(budget) - result.total_cost))
    col4.metric(
        "Startelf", f"{len(picks)} / {sum(position_quota.values())} ({formation})",
    )

    # Anzeige-Spalten: bei Erwartungswert-Zielen die entsprechende Spalte
    # zusätzlich einblenden. Bei Pins die Schloss-Spalte ganz vorne.
    if objective in EXPECTED_OBJECTIVES:
        extra_col = EXPECTED_OBJ_SPECS[objective]["player_col"]
        display_cols = DISPLAY_COLS + [extra_col]
        sort_col = extra_col
    else:
        display_cols = DISPLAY_COLS
        sort_col = "Punkte" if objective == "Punkte" else "Notendurchschnitt"

    show_pin_col = bool(lock_ids)
    if show_pin_col and "Gepinnt" in picks.columns:
        picks = picks.copy()
        picks["🔒"] = picks["Gepinnt"].map({True: "🔒", False: ""})
        display_cols = ["🔒"] + display_cols

    st.divider()
    st.subheader(f"Gewählte Startelf — {formation}")
    pos_cols = st.columns(len(position_quota))
    for col, (pos, quota) in zip(pos_cols, position_quota.items()):
        block = picks[picks["Position"] == pos][display_cols].copy()
        block_sorted = block.sort_values(sort_col, ascending=minimize)
        with col:
            st.markdown(f"**{POSITION_LABEL[pos]}** ({len(block_sorted)}/{quota})")
            fmt: dict[str, object] = {
                "Marktwert": format_eur,
                "Notendurchschnitt": "{:.2f}",
            }
            for c in ("Erwartete Spiele", "Erwartete Tordifferenz"):
                if c in block_sorted.columns:
                    fmt[c] = "{:+.2f}" if c == "Erwartete Tordifferenz" else "{:.2f}"
            st.dataframe(
                block_sorted.style.format(fmt),
                hide_index=True, use_container_width=True,
            )

    st.divider()
    with st.expander("Gesamttabelle / Export"):
        full = picks[["Position"] + display_cols]
        fmt = {"Marktwert": format_eur, "Notendurchschnitt": "{:.2f}"}
        for c in ("Erwartete Spiele", "Erwartete Tordifferenz"):
            if c in full.columns:
                fmt[c] = "{:+.2f}" if c == "Erwartete Tordifferenz" else "{:.2f}"
        st.dataframe(
            full.style.format(fmt),
            hide_index=True, use_container_width=True,
        )
        st.download_button(
            "Kader als CSV herunterladen",
            data=full.to_csv(index=False, sep=";").encode("utf-8"),
            file_name="kicker_kader.csv", mime="text/csv",
        )


# ===================== SIMULATION TAB =====================

def render_simulation() -> None:
    st.subheader("WM 2026 — Monte-Carlo-Simulation")
    st.caption(
        "Rating-Quelle wählen, optional einzelne Werte anpassen, dann Simulation starten. "
        "Die Skala der Ratings beeinflusst die Vorhersagen nicht (z-normalisiert)."
    )

    schedule = cached_load_schedule()

    # ---- Quelle wählen ----
    c_src, _ = st.columns([2, 3])
    with c_src:
        source_label = st.radio(
            "Rating-Quelle",
            options=list(RATING_SOURCES.keys()),
            index=0,
            horizontal=True,
            key="rating_source",
        )
    source = RATING_SOURCES[source_label]
    defaults = cached_default_ratings(source)

    # Wenn Quelle gewechselt wurde, alle Team-Inputs zurücksetzen
    last_source = st.session_state.get("_last_source")
    if last_source != source:
        for team in defaults:
            st.session_state[f"elo_{team}"] = float(defaults[team])
        st.session_state["_last_source"] = source
        # Alte Simulationsergebnisse verwerfen — beziehen sich auf andere Quelle
        st.session_state.pop("sim_results", None)
        st.session_state.pop("custom_team_exp_games", None)
        st.session_state.pop("custom_team_exp_gd", None)

    # Session-State initialisieren (für allerersten Render)
    for team, elo in defaults.items():
        st.session_state.setdefault(f"elo_{team}", float(elo))

    # ---- Gruppen-Grid (4 Spalten × 3 Reihen) ----
    group_letters = list(schedule.groups.keys())
    for row_start in range(0, 12, 4):
        cols = st.columns(4)
        for col, letter in zip(cols, group_letters[row_start : row_start + 4]):
            with col.container(border=True):
                st.markdown(f"#### Gruppe {letter}")
                for team in schedule.groups[letter]:
                    st.number_input(
                        team,
                        min_value=0.0,
                        max_value=5000.0,
                        step=10.0,
                        key=f"elo_{team}",
                        format="%.0f",
                    )

    # ---- Steuerung ----
    st.divider()
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button(f"Auf {source_label}-Default zurücksetzen", use_container_width=True):
            for team, elo in defaults.items():
                st.session_state[f"elo_{team}"] = float(elo)
            st.session_state.pop("sim_results", None)
            st.session_state.pop("custom_team_exp_games", None)
            st.session_state.pop("custom_team_exp_gd", None)
            st.rerun()
    with c2:
        n_runs = st.slider(
            "Anzahl Simulationen",
            min_value=200, max_value=5_000, value=1_000, step=200,
            help=(
                "Lokal ≈ 25 ms/Turnier, Streamlit Cloud (1 vCPU) "
                "deutlich langsamer. 1000 Läufe reichen für robuste "
                "Titel-Wahrscheinlichkeiten."
            ),
        )
    with c3:
        run = st.button("Simulation starten", type="primary", use_container_width=True)

    if run:
        ratings = {team: float(st.session_state[f"elo_{team}"]) for team in defaults}
        progress = st.progress(0.0, text=f"Simuliere {n_runs:,} Turniere…")
        # Nur alle ~1% updaten — sonst dominiert das UI-Roundtrip die Laufzeit.
        update_every = max(1, n_runs // 100)

        def on_step(i: int, total: int) -> None:
            if i == total or i % update_every == 0:
                progress.progress(
                    i / total,
                    text=f"Simuliere Turnier {i:,} / {total:,}",
                )

        df = run_monte_carlo(
            schedule, ratings, n_runs=n_runs, progress_callback=on_step,
        )
        progress.empty()
        st.session_state["sim_results"] = df
        st.session_state["sim_n_runs"] = n_runs
        # Team-Erwartungswerte (DE-Namen) für den Optimizer-Hook bereitstellen.
        st.session_state["custom_team_exp_games"] = mc_to_team_metric_de(df, "exp_games")
        st.session_state["custom_team_exp_gd"] = mc_to_team_metric_de(df, "exp_gd")

    # ---- Ergebnis ----
    if "sim_results" not in st.session_state:
        st.info("Elo-Werte ggf. anpassen und „Simulation starten“ klicken.")
        return

    df = st.session_state["sim_results"].copy()
    n_runs = st.session_state.get("sim_n_runs", "?")

    st.divider()
    st.subheader(f"Ergebnis — {n_runs:,} Simulationen, sortiert nach Titelwahrscheinlichkeit")

    # Titel %, Ø Spiele in den Vordergrund, Details dahinter.
    df = df.rename(columns={
        "team": "Team",
        "rating": "Elo",
        "p_winner": "Titel-Wkt.",
        "exp_games": "Ø Spiele",
        "exp_gd": "Ø TD",
        "p_qualified": "K.o.",
        "p_r16": "R16",
        "p_qf": "QF",
        "p_sf": "SF",
        "p_final": "Finale",
        "p_podium": "Podium",
    })
    df = df[["Team", "Elo", "Titel-Wkt.", "Ø Spiele", "Ø TD",
             "K.o.", "R16", "QF", "SF", "Finale", "Podium"]]

    st.dataframe(
        df.style.format({
            "Elo": "{:.0f}",
            "Titel-Wkt.": "{:.2%}",
            "Ø Spiele": "{:.2f}",
            "Ø TD": "{:+.2f}",
            "K.o.":   "{:.1%}",
            "R16":    "{:.1%}",
            "QF":     "{:.1%}",
            "SF":     "{:.1%}",
            "Finale": "{:.1%}",
            "Podium": "{:.1%}",
        }).background_gradient(subset=["Titel-Wkt."], cmap="Greens")
          .background_gradient(subset=["Ø Spiele"], cmap="Blues")
          .background_gradient(subset=["Ø TD"], cmap="RdYlGn"),
        hide_index=True,
        use_container_width=True,
        height=600,
    )

    csv = df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        "Ergebnis als CSV herunterladen",
        data=csv, file_name="wm_simulation.csv", mime="text/csv",
    )


# ===================== STARTELF-WAHRSCHEINLICHKEITEN =====================

# Canonical store: dict {player_id: probability} unter "probs". Slider-Widgets
# bekommen einen team-skopierten Key (``prob_<team>_<pid>``) — beim Wechsel
# des Teams entstehen frische Widgets, die ihren Anfangswert aus dem
# Canonical-Dict ziehen. Ohne Team-Skopierung würde Streamlit das Widget
# am gleichen Skript-Slot wiederverwenden und den Session-State-Wert
# ignorieren.

def _prob_widget_key(team: str, player_id: str) -> str:
    return f"prob_{team}_{player_id}"


def _ensure_probs_state(player_ids, defaults: dict[str, float]) -> None:
    if "probs" not in st.session_state:
        st.session_state["probs"] = {
            pid: float(defaults.get(pid, 0.0)) for pid in player_ids
        }


def _reset_team_to_defaults(
    team: str, team_ids, defaults: dict[str, float],
) -> None:
    probs = st.session_state["probs"]
    for pid in team_ids:
        probs[pid] = float(defaults.get(pid, 0.0))
        st.session_state.pop(_prob_widget_key(team, pid), None)


def render_probabilities() -> None:
    st.subheader("Startelf-Wahrscheinlichkeiten anpassen")
    st.caption(
        "Pro Team editierbar. Defaults stammen aus den aggregierten "
        "Lineup-Vorhersagen. Änderungen fließen — gemeinsam mit deiner "
        "Simulation — in das Ziel „Erwartete Spiele (Custom)“ ein."
    )

    try:
        players = cached_load_players(str(DATA_DEFAULT))
    except Exception as exc:
        st.error(f"Spieler-CSV konnte nicht geladen werden: {exc}")
        return

    defaults = cached_default_probabilities()
    _ensure_probs_state(players["ID"], defaults)

    # Team-Auswahl + Filter.
    teams = sorted(players["Verein"].unique())
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        team = st.selectbox("Team", options=teams, key="prob_team")
    with c2:
        if st.button("Team auf Default", use_container_width=True):
            team_ids = players.loc[players["Verein"] == team, "ID"]
            _reset_team_to_defaults(team, team_ids, defaults)
            st.rerun()
    with c3:
        if st.button("Alle Teams auf Default", use_container_width=True):
            # Alle Team-Slots im Widget-State löschen, damit beim nächsten
            # Render saubere Widgets aus dem Canonical-Dict initialisiert
            # werden. Wir kennen nur die Widget-Keys des aktuellen Teams
            # (und müssten sie für alle Teams kennen), darum ein Vollscan:
            for key in [k for k in st.session_state if k.startswith("prob_")
                        and k != "prob_team"]:
                del st.session_state[key]
            st.session_state["probs"] = {
                pid: float(defaults.get(pid, 0.0)) for pid in players["ID"]
            }
            st.rerun()

    # Editor je Position.
    team_players = players[players["Verein"] == team].copy()
    if team_players.empty:
        st.info(f"Keine Spieler für {team} gefunden.")
        return

    probs = st.session_state["probs"]

    st.markdown("##### Wahrscheinlichkeiten")
    for pos in ["GOALKEEPER", "DEFENDER", "MIDFIELDER", "FORWARD"]:
        block = team_players[team_players["Position"] == pos]
        if block.empty:
            continue
        with st.container(border=True):
            st.markdown(f"**{POSITION_LABEL[pos]}** ({len(block)})")
            cols = st.columns(2)
            for i, (_, row) in enumerate(block.iterrows()):
                with cols[i % 2]:
                    pid = row["ID"]
                    widget_key = _prob_widget_key(team, pid)
                    # Erst-Render eines Widgets in diesem Team: aus
                    # Canonical-Dict initialisieren. Spätere Renders lesen
                    # den Widget-State direkt.
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = float(probs[pid])
                    st.slider(
                        row["Angezeigter Name"],
                        min_value=0.0, max_value=1.0, step=0.05,
                        key=widget_key,
                        help=f"Default: {defaults.get(pid, 0.0):.2f}",
                    )

    # Widget-Werte zurück in das Canonical-Dict synchronisieren (nur das
    # aktuelle Team — andere Teams behalten ihre zuletzt gesetzten Werte).
    for pid in team_players["ID"]:
        widget_key = _prob_widget_key(team, pid)
        if widget_key in st.session_state:
            probs[pid] = float(st.session_state[widget_key])

    # Custom-Overrides für den Optimizer.
    overrides: dict[str, float] = {}
    for pid, val in probs.items():
        if abs(val - float(defaults.get(pid, 0.0))) > 1e-9:
            overrides[pid] = val
    st.session_state["custom_probabilities"] = overrides

    st.divider()
    st.caption(
        f"Aktuell überschriebene Spieler: **{len(overrides)}** "
        f"(von {len(players)} insgesamt)."
    )
    if not _custom_eg_available():
        st.warning(
            "Noch keine Simulation gelaufen. Wechsle zum Simulator-Tab und "
            "klicke „Simulation starten“, damit „Erwartete Spiele (Custom)“ "
            "im Optimizer verfügbar wird."
        )


# ===================== MAIN =====================

def main() -> None:
    st.set_page_config(page_title="Kicker Manager Optimizer", layout="wide")
    st.title("⚽ Kicker Manager Optimizer")

    with st.sidebar:
        st.markdown("### Modus")
        advanced = st.toggle(
            "Erweiterte Optionen",
            value=False,
            help=(
                "Schaltet Simulator und Startelf-Wahrscheinlichkeiten frei. "
                "Aus deren Kombination wird „Erwartete Spiele (Custom)“ im "
                "Optimizer verfügbar."
            ),
        )
        if advanced:
            st.caption(
                "Custom-Ziel im Optimizer: zuerst auf dem Simulator-Tab eine "
                "Simulation starten, optional auf dem Wahrscheinlichkeiten-"
                "Tab einzelne Spieler anpassen."
            )

    if advanced:
        tab_opt, tab_sim, tab_prob = st.tabs([
            "🧮 Optimizer",
            "🏆 WM-Simulation",
            "👥 Startelf-Wahrscheinlichkeiten",
        ])
        with tab_opt:
            render_optimizer()
        with tab_sim:
            render_simulation()
        with tab_prob:
            render_probabilities()
    else:
        render_optimizer()


if __name__ == "__main__":
    main()
