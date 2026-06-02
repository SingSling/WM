"""Streamlit Dashboard für den Kicker Manager Optimizer + WM-Simulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from optimizer import (
    BUDGET,
    DATA_DEFAULT,
    POSITION_QUOTA,
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


RATING_SOURCES = {
    "Elo (World Football)": "elo",
    "WM-Sieger-Quoten (Buchmacher)": "betting",
}


# ---------- Helpers ----------

def format_eur(value: int) -> str:
    return f"{value:,} €".replace(",", ".")


# ===================== OPTIMIZER TAB =====================

def render_optimizer() -> None:
    st.subheader("Kader-Optimierung")

    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 2])
        with c1:
            data_path = st.text_input("CSV-Pfad", value=str(DATA_DEFAULT), key="opt_csv")
        with c2:
            budget = st.number_input(
                "Budget (€)", min_value=10_000_000, max_value=200_000_000,
                value=BUDGET, step=1_000_000, key="opt_budget",
            )
        with c3:
            objective = st.selectbox(
                "Zielgröße", options=["Punkte", "Notendurchschnitt"], index=0, key="opt_obj",
            )

        c4, c5 = st.columns([3, 1])
        with c4:
            if objective == "Punkte":
                minimize = False
                exclude_zero = st.checkbox(
                    "Spieler mit 0 Punkten ausschließen", value=False, key="opt_zero_p",
                )
            else:
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
        with c5:
            run = st.button("Optimieren", type="primary", use_container_width=True)

    try:
        players = cached_load_players(data_path)
    except Exception as exc:
        st.error(f"CSV konnte nicht geladen werden: {exc}")
        return
    st.caption(f"{len(players)} Spieler geladen aus `{data_path}`")

    if not run:
        st.info("Einstellungen wählen und „Optimieren“ klicken.")
        return

    with st.spinner("Solver läuft…"):
        try:
            result = optimize(
                players, objective_col=objective, minimize=minimize,
                exclude_zero_objective=exclude_zero, budget=int(budget),
            )
        except Exception as exc:
            st.error(f"Optimierung fehlgeschlagen: {exc}")
            return

    picks = result.picks
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"Σ {objective}", f"{result.objective_value:.2f}")
    col2.metric("Gesamtkosten", format_eur(result.total_cost))
    col3.metric("Restbudget", format_eur(int(budget) - result.total_cost))
    col4.metric("Kadergröße", f"{len(picks)} / {sum(POSITION_QUOTA.values())}")

    st.divider()
    st.subheader("Gewählter Kader")
    pos_cols = st.columns(len(POSITION_QUOTA))
    for col, (pos, quota) in zip(pos_cols, POSITION_QUOTA.items()):
        block = picks[picks["Position"] == pos][DISPLAY_COLS].copy()
        block_sorted = block.sort_values(
            "Punkte" if objective == "Punkte" else "Notendurchschnitt",
            ascending=minimize,
        )
        with col:
            st.markdown(f"**{POSITION_LABEL[pos]}** ({len(block_sorted)}/{quota})")
            st.dataframe(
                block_sorted.style.format(
                    {"Marktwert": format_eur, "Notendurchschnitt": "{:.2f}"}
                ),
                hide_index=True, use_container_width=True,
            )

    st.divider()
    with st.expander("Gesamttabelle / Export"):
        full = picks[["Position"] + DISPLAY_COLS]
        st.dataframe(
            full.style.format({"Marktwert": format_eur, "Notendurchschnitt": "{:.2f}"}),
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
            st.rerun()
    with c2:
        n_runs = st.slider(
            "Anzahl Simulationen",
            min_value=500, max_value=10_000, value=2_000, step=500,
            help="2000 ≈ 50 s · 5000 ≈ 2 min · 10000 ≈ 4 min",
        )
    with c3:
        run = st.button("Simulation starten", type="primary", use_container_width=True)

    if run:
        ratings = {team: float(st.session_state[f"elo_{team}"]) for team in defaults}
        with st.spinner(f"Simuliere {n_runs:,} Turniere… (~{n_runs * 0.025:.0f} s)"):
            df = run_monte_carlo(schedule, ratings, n_runs=n_runs)
        st.session_state["sim_results"] = df
        st.session_state["sim_n_runs"] = n_runs

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
        "p_qualified": "K.o.",
        "p_r16": "R16",
        "p_qf": "QF",
        "p_sf": "SF",
        "p_final": "Finale",
        "p_podium": "Podium",
    })
    df = df[["Team", "Elo", "Titel-Wkt.", "Ø Spiele",
             "K.o.", "R16", "QF", "SF", "Finale", "Podium"]]

    st.dataframe(
        df.style.format({
            "Elo": "{:.0f}",
            "Titel-Wkt.": "{:.2%}",
            "Ø Spiele": "{:.2f}",
            "K.o.":   "{:.1%}",
            "R16":    "{:.1%}",
            "QF":     "{:.1%}",
            "SF":     "{:.1%}",
            "Finale": "{:.1%}",
            "Podium": "{:.1%}",
        }).background_gradient(subset=["Titel-Wkt."], cmap="Greens")
          .background_gradient(subset=["Ø Spiele"], cmap="Blues"),
        hide_index=True,
        use_container_width=True,
        height=600,
    )

    csv = df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
        "Ergebnis als CSV herunterladen",
        data=csv, file_name="wm_simulation.csv", mime="text/csv",
    )


# ===================== MAIN =====================

def main() -> None:
    st.set_page_config(page_title="Kicker Manager Optimizer", layout="wide")
    st.title("⚽ Kicker Manager Optimizer")

    tab_opt, tab_sim = st.tabs(["🧮 Optimizer", "🏆 WM-Simulation"])
    with tab_opt:
        render_optimizer()
    with tab_sim:
        render_simulation()


if __name__ == "__main__":
    main()
