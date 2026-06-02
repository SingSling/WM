"""Team-Stärken für die WM-2026-Simulation.

Quellen (Argument ``source`` in ``load_ratings``):

* ``"elo"`` — Default. World Football Elo von eloratings.net, gecached in
  ``Data/elo_cache.json``, optionale partielle Overrides in
  ``Data/ratings_override.json``.

* ``"betting"`` — Aus Buchmacher-Quoten in ``Data/betting_odds.json``
  abgeleitet (American Odds für Gruppensieg). Innerhalb der Gruppe folgt
  das Ranking den Quoten (zentrierte log-Probs), die Gruppen-Mittelwerte
  und die Gesamtskala werden an Elo angeglichen — damit die Zahlen optisch
  vergleichbar bleiben und Cross-Group-Vergleiche sinnvoll funktionieren.

* ``"custom"`` — ``Data/ratings_custom.json`` ersetzt alles. Beliebige
  Skala — der Simulator normalisiert intern via z-Scores.
"""

from __future__ import annotations

import json
import math
import statistics
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
CACHE_PATH = DATA_DIR / "elo_cache.json"
OVERRIDE_PATH = DATA_DIR / "ratings_override.json"
CUSTOM_PATH = DATA_DIR / "ratings_custom.json"
BETTING_PATH = DATA_DIR / "betting_odds.json"
ELO_URL = "https://www.eloratings.net/World.tsv"

# Mapping von unseren Team-Namen auf die Country-Codes bei eloratings.net.
# Achtung: EN/SQ/WA für englische Heimnationen statt ISO-Codes.
TEAM_TO_CODE = {
    "Mexico": "MX", "South Africa": "ZA", "South Korea": "KR", "Czechia": "CZ",
    "Canada": "CA", "Bosnia-Herzegovina": "BA", "Qatar": "QA", "Switzerland": "CH",
    "Brazil": "BR", "Morocco": "MA", "Haiti": "HT", "Scotland": "SQ",
    "USA": "US", "Paraguay": "PY", "Australia": "AU", "Turkey": "TR",
    "Germany": "DE", "Curacao": "CW", "Ivory Coast": "CI", "Ecuador": "EC",
    "Netherlands": "NL", "Japan": "JP", "Sweden": "SE", "Tunisia": "TN",
    "Belgium": "BE", "Egypt": "EG", "Iran": "IR", "New Zealand": "NZ",
    "Spain": "ES", "Cape Verde": "CV", "Saudi Arabia": "SA", "Uruguay": "UY",
    "France": "FR", "Senegal": "SN", "Iraq": "IQ", "Norway": "NO",
    "Argentina": "AR", "Algeria": "DZ", "Austria": "AT", "Jordan": "JO",
    "Portugal": "PT", "Congo DR": "CD", "Uzbekistan": "UZ", "Colombia": "CO",
    "England": "EN", "Croatia": "HR", "Ghana": "GH", "Panama": "PA",
}


def _fetch_world_elo() -> dict[str, float]:
    """Lädt die World.tsv und gibt {code: elo} zurück."""
    req = urllib.request.Request(ELO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        text = resp.read().decode("utf-8")
    codes: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        try:
            codes[parts[2]] = float(parts[3])
        except (ValueError, IndexError):
            continue
    return codes


def _load_elo_with_overrides(refresh: bool) -> dict[str, float]:
    if refresh or not CACHE_PATH.exists():
        codes = _fetch_world_elo()
        CACHE_PATH.write_text(json.dumps(codes, indent=2), encoding="utf-8")
    else:
        codes = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    ratings: dict[str, float] = {}
    missing_codes: list[str] = []
    for team, code in TEAM_TO_CODE.items():
        if code in codes:
            ratings[team] = codes[code]
        else:
            missing_codes.append(f"{team} ({code})")
    if missing_codes:
        raise KeyError(f"Elo-Codes nicht gefunden: {missing_codes}")

    if OVERRIDE_PATH.exists():
        overrides = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))
        for team, val in overrides.items():
            if team not in ratings:
                raise KeyError(f"Override für unbekanntes Team: {team}")
            ratings[team] = float(val)
    return ratings


def _load_custom() -> dict[str, float]:
    custom_raw = json.loads(CUSTOM_PATH.read_text(encoding="utf-8"))
    ratings = {k: float(v) for k, v in custom_raw.items()}
    missing = set(TEAM_TO_CODE) - ratings.keys()
    extra = ratings.keys() - set(TEAM_TO_CODE)
    if missing:
        raise KeyError(
            f"ratings_custom.json: fehlende Teams ({len(missing)}): {sorted(missing)}"
        )
    if extra:
        raise KeyError(f"ratings_custom.json: unbekannte Teams: {sorted(extra)}")
    return ratings


# ---------- Betting-Odds-basierte Ratings ----------

def american_odds_to_prob(odds: float) -> float:
    """American Odds → implizite Wahrscheinlichkeit (mit Buchmacher-Marge)."""
    if odds < 0:
        return -odds / (-odds + 100.0)
    return 100.0 / (odds + 100.0)


def _load_betting_odds() -> dict[str, float]:
    raw = json.loads(BETTING_PATH.read_text(encoding="utf-8"))
    odds = {k: float(v) for k, v in raw["odds"].items()}
    missing = set(TEAM_TO_CODE) - odds.keys()
    extra = odds.keys() - set(TEAM_TO_CODE)
    if missing:
        raise KeyError(f"betting_odds.json: fehlende Teams: {sorted(missing)}")
    if extra:
        raise KeyError(f"betting_odds.json: unbekannte Teams: {sorted(extra)}")
    return odds


def _compute_betting_ratings(elo_anchor: dict[str, float]) -> dict[str, float]:
    """Leitet Ratings aus WM-Sieger-Quoten ab.

    Die Quoten gelten für den Gewinn der WM und sind daher schon zwischen
    den Gruppen vergleichbar. Verarbeitung:

    1. American Odds → implizite Wahrscheinlichkeit (mit Buchmacher-Marge,
       nicht weiter normalisiert — die *relativen* Werte über alle 48 Teams
       hinweg sind das Signal).
    2. log(p) als Rating (Bradley-Terry-/Logit-artig).
    3. Global auf Elo-Skala umrechnen (gleiche Mittelwert und
       Standardabweichung), nur damit die Zahlen lesbar bleiben — der
       Simulator z-normalisiert intern, die Skala beeinflusst Vorhersagen nicht.

    ``elo_anchor`` wird *nur* für Mittelwert/Std der Display-Skala
    verwendet, nicht für inhaltliche Mischung mit den Buchmacher-Daten.
    """
    odds = _load_betting_odds()

    # 1) Wahrscheinlichkeiten (kein per-Group-Normalisieren)
    probs = {t: american_odds_to_prob(o) for t, o in odds.items()}

    # 2) log-Probs als Rohrating
    raw = {t: math.log(p) for t, p in probs.items()}

    # 3) Linear auf Elo-Skala mappen
    elo_vals = list(elo_anchor.values())
    elo_mean = statistics.mean(elo_vals)
    elo_std = statistics.stdev(elo_vals)

    raw_vals = list(raw.values())
    raw_mean = statistics.mean(raw_vals)
    raw_std = statistics.stdev(raw_vals) or 1.0

    scale = elo_std / raw_std
    return {t: elo_mean + (r - raw_mean) * scale for t, r in raw.items()}


# ---------- Öffentliche API ----------

def load_ratings(source: str = "elo", refresh: bool = False) -> dict[str, float]:
    """Liefert ``{team_name: rating}`` für alle 48 WM-Teams.

    Parameter
    ---------
    source : ``"elo"`` (Default), ``"betting"`` oder ``"custom"``.
    refresh : Bei Elo: Cache verwerfen und neu aus dem Web laden.
    """
    if source == "custom":
        if not CUSTOM_PATH.exists():
            raise FileNotFoundError(f"{CUSTOM_PATH} fehlt — siehe write_custom_template()")
        return _load_custom()

    if source == "elo":
        return _load_elo_with_overrides(refresh)

    if source == "betting":
        # Anker via Elo (für Cross-Group-Kalibrierung)
        elo = _load_elo_with_overrides(refresh)
        return _compute_betting_ratings(elo_anchor=elo)

    raise ValueError(f"Unbekannte source: {source!r}. Erlaubt: elo, betting, custom")


def write_custom_template(path: Path = CUSTOM_PATH, scale: tuple[float, float] = (1.0, 10.0)) -> None:
    """Schreibt eine Vorlage mit allen 48 Teams auf der angegebenen Skala (lo, hi).

    Default-Werte werden auf den Skalenmittelpunkt gesetzt — der User passt
    danach an. Existierende Datei wird *nicht* überschrieben.
    """
    if path.exists():
        raise FileExistsError(f"{path} existiert bereits — nicht überschrieben.")
    mid = (scale[0] + scale[1]) / 2
    data = {team: mid for team in TEAM_TO_CODE}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["elo", "betting", "custom"], default="elo")
    args = parser.parse_args()
    r = load_ratings(source=args.source)
    print(f"Quelle: {args.source}  ({len(r)} Teams)\n")
    for team, elo in sorted(r.items(), key=lambda x: -x[1]):
        print(f"  {team:25s} {elo:7.1f}")
