"""Berechnet Startaufstellungs-Wahrscheinlichkeiten pro Spieler.

Eingabe:
* ``Data/predicted_lineups.json`` — gesammelte Vorhersagen aus mehreren Quellen
  (Bulinews, RotoWire, fifaworldcup.live, ESPN clean, ESPN images).
* ``Data/players-se-k01012026.csv`` — offizielle Spielerliste (deutsche
  Teamnamen).

Ausgabe:
* ``Data/lineup_probabilities.csv`` — pro CSV-Spieler:
  ``probability = matches / sources_for_team``.

Name-Matching: akzent-/zeichen-normalisiert, Nachname-basiert mit Vornamen-
Tiebreaker. Eine Vorhersagename gilt als gematcht, wenn ein eindeutiger
CSV-Spieler im selben Team gefunden wird.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
CSV_PATH = DATA_DIR / "players-se-k01012026.csv"
LINEUPS_PATH = DATA_DIR / "predicted_lineups.json"
OUT_CSV = DATA_DIR / "lineup_probabilities.csv"
UNMATCHED_LOG = DATA_DIR / "lineup_unmatched.txt"

# Englisch (Lineup-JSON) -> Deutsch (CSV "Verein"-Spalte).
TEAM_EN_TO_DE = {
    "Mexico": "Mexiko",
    "South Africa": "Südafrika",
    "South Korea": "Südkorea",
    "Czech Republic": "Tschechien",
    "Canada": "Kanada",
    "Bosnia": "Bosnien-Herzegowina",
    "Qatar": "Katar",
    "Switzerland": "Schweiz",
    "Brazil": "Brasilien",
    "Morocco": "Marokko",
    "Haiti": "Haiti",
    "Scotland": "Schottland",
    "USA": "USA",
    "Paraguay": "Paraguay",
    "Australia": "Australien",
    "Turkey": "Türkei",
    "Germany": "Deutschland",
    "Curacao": "Curacao",
    "Ivory Coast": "Elfenbeinküste",
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
    "Austria": "Österreich",
    "Jordan": "Jordanien",
    "Argentina": "Argentinien",
    "Algeria": "Algerien",
    "Portugal": "Portugal",
    "DR Congo": "DR Kongo",
    "Uzbekistan": "Usbekistan",
    "Colombia": "Kolumbien",
    "England": "England",
    "Croatia": "Kroatien",
    "Ghana": "Ghana",
    "Panama": "Panama",
}


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize(s: str) -> str:
    """Lowercase, akzent-frei, alphanumerische Tokens nur."""
    s = strip_accents(s).lower()
    s = s.replace("'", " ").replace("-", " ").replace(".", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Häufige Vornamen-Initialen, die vor dem eigentlichen Namen stehen können.
INITIAL_RE = re.compile(r"^[a-z]\b\s*")


def tokens(s: str) -> list[str]:
    return normalize(s).split()


def player_aliases(first: str, last: str, display: str) -> set[str]:
    """Alle Schreibvarianten, mit denen ein CSV-Spieler matchen kann."""
    aliases: set[str] = set()
    full_tokens = tokens(f"{first} {last}")
    last_tokens = tokens(last)
    display_tokens = tokens(display)
    first_tokens = tokens(first)

    if full_tokens:
        aliases.add(" ".join(full_tokens))
    if last_tokens:
        aliases.add(" ".join(last_tokens))
        # Jedes einzelne Token im Nachnamen als möglichen Match (z.B.
        # "Araujo Vilches" → sowohl "araujo" als auch "vilches").
        for t in last_tokens:
            if len(t) >= 3:
                aliases.add(t)
    if display_tokens:
        aliases.add(" ".join(display_tokens))
        for t in display_tokens:
            if len(t) >= 3:
                aliases.add(t)
    if first_tokens and last_tokens:
        # "I. Saibari"-Stil: Initial + Nachname.
        aliases.add(f"{first_tokens[0][0]} {last_tokens[-1]}")
        aliases.add(f"{first_tokens[0]} {last_tokens[-1]}")
        # "Surname Firstname"-Reihenfolge (asiatischer Stil): in den Aliases
        # legen wir auch "last first" und "last firstparts" ab.
        aliases.add(f"{last_tokens[-1]} {first_tokens[0]}")
        aliases.add(f"{last_tokens[-1]} {' '.join(first_tokens)}")
    return aliases


def last_name_tokens_set(last: str) -> set[str]:
    return set(t for t in tokens(last) if len(t) >= 2)


def load_players() -> list[dict]:
    players: list[dict] = []
    with CSV_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            players.append({
                "id": row["ID"],
                "first": row["Vorname"],
                "last": row["Nachname"],
                "display": row["Angezeigter Name"],
                "display_short": row["Angezeigter Name (kurz)"],
                "team_de": row["Verein"],
                "position": row["Position"],
                "market_value": row["Marktwert"],
                "points": row["Punkte"],
                "avg_grade": row["Notendurchschnitt"],
            })
    return players


def build_team_index(players: list[dict]) -> dict[str, list[dict]]:
    """{team_de: [player_records]}"""
    idx: dict[str, list[dict]] = {}
    for p in players:
        idx.setdefault(p["team_de"], []).append(p)
    return idx


def match_prediction(pred_name: str, candidates: list[dict]) -> dict | None:
    """Findet den eindeutigen CSV-Spieler für eine Vorhersage-Schreibweise.

    Mehrstufige Strategie:
    1. Exakte normalisierte Match auf irgend einem Alias.
    2. Letztes Token des Pred-Namens == letztes Token irgendeines Alias.
    3. Erstes Token == letztes Token (für Single-Name Spieler wie "Casemiro").

    Bei Mehrdeutigkeit (mehrere Treffer) Filter nach gemeinsamem Vornamen-
    Token. Wenn dann immer noch mehrdeutig -> None (kein Match).
    """
    pred_tokens = tokens(pred_name)
    if not pred_tokens:
        return None
    pred_norm = " ".join(pred_tokens)
    pred_last = pred_tokens[-1]
    pred_first = pred_tokens[0]

    # Stufe 1: exakter Alias-Match.
    exact: list[dict] = []
    for c in candidates:
        if pred_norm in c["_aliases"]:
            exact.append(c)
    if len(exact) == 1:
        return exact[0]

    # Stufe 2: Nachname-Match. pred_last muss in irgendeinem Token des
    # CSV-Nachnamens vorkommen (deckt "Araujo Vilches" und ähnliche ab).
    last_match: list[dict] = []
    for c in candidates:
        if pred_last in last_name_tokens_set(c["last"]):
            last_match.append(c)
    if len(last_match) == 1:
        return last_match[0]

    # Bei Mehrdeutigkeit: Filter nach Vornamen-Übereinstimmung.
    if len(last_match) > 1:
        candidates_first = [
            c for c in last_match
            if tokens(c["first"]) and (
                tokens(c["first"])[0] == pred_first
                or (len(pred_first) == 1 and tokens(c["first"])[0].startswith(pred_first))
                or (len(pred_tokens) > 1 and pred_tokens[0] in tokens(c["first"]))
                # auch: voller Vorname als zweites Token des Pred-Namens
                or any(t in tokens(c["first"]) for t in pred_tokens[1:])
            )
        ]
        if len(candidates_first) == 1:
            return candidates_first[0]

    # Stufe 2b: Reverse-Order (Asian-Style "Surname First"):
    # pred_first ist möglicherweise der Nachname.
    reverse_match: list[dict] = []
    for c in candidates:
        if pred_first in last_name_tokens_set(c["last"]):
            reverse_match.append(c)
    if len(reverse_match) == 1:
        return reverse_match[0]
    if len(reverse_match) > 1 and len(pred_tokens) > 1:
        # Disambiguiere via Vorname-Match auf restliche Pred-Tokens.
        rest = pred_tokens[1:]
        first_match = [
            c for c in reverse_match
            if any(t in tokens(c["first"]) for t in rest)
        ]
        if len(first_match) == 1:
            return first_match[0]

    # Stufe 3: Single-Name (z.B. "Casemiro", "Zizo", "Vinicius"): Pred-Name als
    # gesamtes Match gegen Vornamen, Nachnamen oder Anzeigenamen.
    if len(pred_tokens) == 1:
        single: list[dict] = []
        for c in candidates:
            for alias in c["_aliases"]:
                if alias == pred_norm:
                    single.append(c)
                    break
            else:
                # Auch Display-Name (kurz) prüfen
                if pred_norm == normalize(c["display_short"]):
                    single.append(c)
        # _aliases enthält display_short bereits, also keine Doppelung nötig.
        # Falls genau einer mit irgendeiner Token-Kombi passt:
        if len(single) == 1:
            return single[0]

    # Stufe 4: substring-fallback nur, wenn pred recht spezifisch (>=5 chars).
    if len(pred_last) >= 5:
        subs: list[dict] = []
        for c in candidates:
            last_norm = normalize(c["last"])
            disp_norm = normalize(c["display"])
            if pred_last in last_norm or pred_last in disp_norm:
                subs.append(c)
        if len(subs) == 1:
            return subs[0]

    return None


def main() -> None:
    players = load_players()
    team_index = build_team_index(players)
    lineups = json.loads(LINEUPS_PATH.read_text(encoding="utf-8"))["lineups"]

    # Aliases pro Spieler vorberechnen.
    for p in players:
        p["_aliases"] = player_aliases(p["first"], p["last"], p["display"])
        p["_aliases"].add(normalize(p["display_short"]))

    # Init: appearances counter pro player-ID.
    appearances: dict[str, int] = {p["id"]: 0 for p in players}
    sources_for_team: dict[str, int] = {}  # team_de -> Anzahl Quellen
    unmatched: list[tuple[str, str, str]] = []  # (team, source, pred_name)

    for team_en, sources in lineups.items():
        team_de = TEAM_EN_TO_DE.get(team_en)
        if team_de is None:
            print(f"WARN: Kein DE-Mapping für '{team_en}'")
            continue
        candidates = team_index.get(team_de, [])
        if not candidates:
            print(f"WARN: Keine CSV-Spieler für Team '{team_de}'")
            continue
        sources_for_team[team_de] = len(sources)
        for source_name, names in sources.items():
            for pred in names:
                matched = match_prediction(pred, candidates)
                if matched is None:
                    unmatched.append((team_de, source_name, pred))
                else:
                    appearances[matched["id"]] += 1

    # Output CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "ID", "Vorname", "Nachname", "Angezeigter Name", "Team",
            "Position", "Marktwert", "Appearances", "Sources",
            "Probability",
        ])
        for p in players:
            srcs = sources_for_team.get(p["team_de"], 0)
            apps = appearances[p["id"]]
            prob = apps / srcs if srcs else 0.0
            writer.writerow([
                p["id"], p["first"], p["last"], p["display"], p["team_de"],
                p["position"], p["market_value"], apps, srcs,
                f"{prob:.3f}",
            ])

    # Unmatched-Log
    with UNMATCHED_LOG.open("w", encoding="utf-8") as f:
        f.write(f"# Unmatched predictions: {len(unmatched)}\n")
        for team, src, pred in unmatched:
            f.write(f"{team} | {src} | {pred}\n")

    # Kurze Zusammenfassung
    total_preds = sum(len(srcs) * 1 for team in lineups.values() for srcs in team.values())
    total_player_slots = sum(len(names) for team in lineups.values() for names in team.values())
    matched_count = total_player_slots - len(unmatched)
    print(f"Quellen pro Team: min={min(sources_for_team.values())}, "
          f"max={max(sources_for_team.values())}, "
          f"avg={sum(sources_for_team.values())/len(sources_for_team):.1f}")
    print(f"Vorhersage-Slots gesamt: {total_player_slots}")
    print(f"Davon gematcht: {matched_count} ({matched_count/total_player_slots*100:.1f}%)")
    print(f"Unmatched: {len(unmatched)} -> {UNMATCHED_LOG}")
    print(f"Output: {OUT_CSV}")

    # Top 20 nach Wahrscheinlichkeit (nur prob > 0)
    rows = []
    for p in players:
        srcs = sources_for_team.get(p["team_de"], 0)
        apps = appearances[p["id"]]
        if srcs and apps:
            rows.append((apps / srcs, apps, srcs, p))
    rows.sort(key=lambda r: (-r[0], -r[1], r[3]["last"]))
    print("\nTop 30 Spieler nach Start-Wahrscheinlichkeit:")
    for prob, apps, srcs, p in rows[:30]:
        print(f"  {prob:.2f}  ({apps}/{srcs})  {p['display']:30s} {p['team_de']}")


if __name__ == "__main__":
    main()
