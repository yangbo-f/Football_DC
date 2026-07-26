from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


MATCH_COLUMNS = [
    "competition",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "neutral_site",
]

OPTIONAL_MATCH_COLUMNS = [
    "stage",
    "round",
    "score_basis",
    "decided_by_penalties",
    "winner",
    "notes",
    "source_file",
    "match_importance",
    "prediction_available_at",
]

STANDARD_MATCH_COLUMNS = MATCH_COLUMNS + OPTIONAL_MATCH_COLUMNS

FOOTBALL_DATA_COLUMN_MAP = {
    "Div": "competition",
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
}

FOOTBALL_DATA_COMPETITION_MAP = {
    "E0": "EPL",
}

TEAM_NAME_ALIASES = {
    "Cape Verde": "Cabo Verde",
    "Curacao": "Curaçao",
    "Czech Republic": "Czechia",
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "Türki̇ye": "Turkey",
    "Türkiye": "Turkey",
    "USA": "United States",
}

SCORE_BASIS_MAP = {
    "": "FT90",
    "FT": "FT90",
    "FULL_TIME": "FT90",
    "FULL TIME": "FT90",
    "FULLTIME": "FT90",
    "90": "FT90",
    "90MIN": "FT90",
    "90_MIN": "FT90",
    "FT90": "FT90",
}

TRAINING_COMPETITION_GROUPS = {
    "WorldCup": ["WorldCup", "WorldCupQualifiers"],
    "WomenWorldCup": ["WomenWorldCup", "WomenWorldCupQualifiers"],
}


@dataclass(frozen=True)
class CompetitionConfig:
    code: str
    label: str
    default_neutral_site: bool = False


COMPETITIONS: Dict[str, CompetitionConfig] = {
    "EPL": CompetitionConfig(code="EPL", label="英超", default_neutral_site=False),
    "CSL": CompetitionConfig(code="CSL", label="中超", default_neutral_site=False),
    "WorldCup": CompetitionConfig(code="WorldCup", label="世界杯", default_neutral_site=True),
    "WorldCupQualifiers": CompetitionConfig(code="WorldCupQualifiers", label="世界杯预选赛", default_neutral_site=False),
    "WomenWorldCup": CompetitionConfig(code="WomenWorldCup", label="女足世界杯", default_neutral_site=True),
    "WomenWorldCupQualifiers": CompetitionConfig(code="WomenWorldCupQualifiers", label="女足世界杯预选赛", default_neutral_site=False),
}


def load_matches(path: str | Path, competition: Optional[str] = None, season: Optional[str] = None) -> pd.DataFrame:
    return normalize_matches(pd.read_csv(path), competition=competition, season=season)


def normalize_matches(matches: pd.DataFrame, competition: Optional[str] = None, season: Optional[str] = None) -> pd.DataFrame:
    normalized = matches.rename(columns=FOOTBALL_DATA_COLUMN_MAP).copy()

    if competition is not None:
        normalized["competition"] = competition
    elif "competition" not in normalized.columns:
        normalized["competition"] = "Unknown"

    if season is not None:
        normalized["season"] = season
    elif "season" not in normalized.columns:
        normalized["season"] = ""

    if "neutral_site" not in normalized.columns:
        normalized["neutral_site"] = normalized["competition"].map(default_neutral_site)

    for optional in OPTIONAL_MATCH_COLUMNS:
        if optional not in normalized.columns:
            normalized[optional] = default_optional_value(optional)

    missing = set(MATCH_COLUMNS) - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required match columns: {sorted(missing)}")

    normalized = normalized[STANDARD_MATCH_COLUMNS].copy()
    normalized["competition"] = normalized["competition"].replace(FOOTBALL_DATA_COMPETITION_MAP)
    normalized["home_team"] = normalized["home_team"].map(normalize_team_name)
    normalized["away_team"] = normalized["away_team"].map(normalize_team_name)
    if "winner" in normalized.columns:
        normalized["winner"] = normalized["winner"].map(normalize_team_name)
    normalized["date"] = parse_dates(normalized["date"])
    normalized["home_goals"] = pd.to_numeric(normalized["home_goals"], errors="coerce")
    normalized["away_goals"] = pd.to_numeric(normalized["away_goals"], errors="coerce")
    normalized["neutral_site"] = normalized["neutral_site"].map(normalize_bool)
    normalized["score_basis"] = normalized["score_basis"].map(normalize_score_basis)
    normalized["decided_by_penalties"] = normalized["decided_by_penalties"].fillna(False).astype(bool)
    normalized["source_file"] = normalized["source_file"].fillna("").astype(str)
    normalized["match_importance"] = normalize_match_importance(normalized)
    normalized["prediction_available_at"] = parse_prediction_available_at(normalized)
    normalized = normalized.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    normalized["home_goals"] = normalized["home_goals"].astype(int)
    normalized["away_goals"] = normalized["away_goals"].astype(int)
    return normalized.sort_values("date").reset_index(drop=True)


def default_optional_value(column: str):
    if column == "score_basis":
        return "FT90"
    if column == "decided_by_penalties":
        return False
    if column == "match_importance":
        return ""
    return ""


def normalize_match_importance(matches: pd.DataFrame) -> pd.Series:
    explicit = pd.to_numeric(matches["match_importance"], errors="coerce")
    inferred = matches.apply(infer_match_importance, axis=1)
    return explicit.fillna(inferred).astype(float).clip(lower=0.0, upper=1.5)


def infer_match_importance(row: pd.Series) -> float:
    competition = str(row.get("competition", "")).strip()
    stage = str(row.get("stage", "")).strip().lower()
    round_name = str(row.get("round", "")).strip().lower()

    if competition in {"WorldCup", "WomenWorldCup"}:
        if "knockout" in stage or round_name in {"round of 32", "round of 16", "quarterfinals", "semifinals", "final"}:
            return 1.0
        if "group" in stage or round_name == "group stage":
            return 0.95
        return 0.9
    if competition in {"WorldCupQualifiers", "WomenWorldCupQualifiers"}:
        return 0.75
    if competition in {"EPL", "CSL"}:
        return 0.7
    return 0.5


def parse_prediction_available_at(matches: pd.DataFrame) -> pd.Series:
    parsed = pd.to_datetime(matches["prediction_available_at"], errors="coerce")
    return parsed.fillna(matches["date"])


def normalize_score_basis(value) -> str:
    if pd.isna(value):
        return "FT90"
    text = str(value).strip()
    key = text.upper().replace("-", "_")
    return SCORE_BASIS_MAP.get(key, text.upper())


def normalize_team_name(value):
    if pd.isna(value):
        return value
    text = str(value).strip()
    if not text:
        return text
    return TEAM_NAME_ALIASES.get(text, text)


def normalize_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def default_neutral_site(competition: str) -> bool:
    return COMPETITIONS.get(str(competition), CompetitionConfig(str(competition), str(competition))).default_neutral_site


def parse_dates(dates: pd.Series) -> pd.Series:
    as_text = dates.astype(str)
    day_first = as_text.str.contains("/").any()
    return pd.to_datetime(dates, dayfirst=day_first, errors="coerce")


def filter_competition(matches: pd.DataFrame, competition: str) -> pd.DataFrame:
    members = TRAINING_COMPETITION_GROUPS.get(competition, [competition])
    return matches[matches["competition"].astype(str).isin(members)].copy().reset_index(drop=True)


def filter_training_matches(matches: pd.DataFrame, competition: str) -> pd.DataFrame:
    filtered = filter_competition(matches, competition)
    if "score_basis" not in filtered.columns:
        return filtered
    basis = filtered["score_basis"].fillna("FT90").replace("", "FT90").astype(str).str.upper()
    return filtered[basis == "FT90"].copy().reset_index(drop=True)


def teams_for_competition(matches: pd.DataFrame, competition: str) -> list[str]:
    filtered = filter_competition(matches, competition)
    teams = set(filtered["home_team"]) | set(filtered["away_team"])
    return sorted(str(team) for team in teams)


def combine_match_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [frame for frame in frames if frame is not None and not frame.empty]
    if not usable:
        return pd.DataFrame(columns=STANDARD_MATCH_COLUMNS)
    return pd.concat(usable, ignore_index=True).sort_values("date").reset_index(drop=True)
