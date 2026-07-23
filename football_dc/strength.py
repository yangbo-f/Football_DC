from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Iterable

import pandas as pd

from .dixon_coles import Prediction, prediction_from_expectations


CONFEDERATION_BASE_RATING = {
    "UEFA": 1560.0,
    "CONMEBOL": 1555.0,
    "CAF": 1485.0,
    "CONCACAF": 1475.0,
    "AFC": 1465.0,
    "OFC": 1375.0,
}


TEAM_RATING_OVERRIDES = {
    "Argentina": 1660.0,
    "France": 1655.0,
    "Spain": 1645.0,
    "England": 1640.0,
    "Brazil": 1635.0,
    "Portugal": 1625.0,
    "Netherlands": 1615.0,
    "Belgium": 1605.0,
    "Germany": 1600.0,
    "Italy": 1595.0,
    "Uruguay": 1590.0,
    "Croatia": 1580.0,
    "Colombia": 1580.0,
    "Morocco": 1575.0,
    "Switzerland": 1570.0,
    "Austria": 1565.0,
    "Denmark": 1565.0,
    "Mexico": 1550.0,
    "United States": 1545.0,
    "Japan": 1540.0,
    "Senegal": 1535.0,
    "Nigeria": 1530.0,
    "Egypt": 1525.0,
    "Iran": 1520.0,
    "South Korea": 1515.0,
    "Tunisia": 1515.0,
    "Australia": 1500.0,
    "Saudi Arabia": 1490.0,
    "Qatar": 1485.0,
    "South Africa": 1480.0,
    "China PR": 1435.0,
    "New Zealand": 1450.0,
}


CONFEDERATION_TEAMS = {
    "AFC": {
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia", "China PR",
        "Guam", "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan", "Kuwait",
        "Kyrgyzstan", "Laos", "Lebanon", "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar",
        "Nepal", "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
        "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan",
        "Thailand", "Timor-Leste", "Turkmenistan", "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
    },
    "CAF": {
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cameroon", "Cabo Verde",
        "Central African Republic", "Chad", "Comoros", "Congo", "DR Congo", "Djibouti", "Egypt",
        "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
        "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi",
        "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
        "Rwanda", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan",
        "Sudan", "São Tomé and Príncipe", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe",
    },
    "CONCACAF": {
        "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize", "Bermuda",
        "British Virgin Islands", "Canada", "Cayman Islands", "Costa Rica", "Cuba", "Curaçao",
        "Dominica", "Dominican Republic", "El Salvador", "Grenada", "Guatemala", "Guyana", "Haiti",
        "Honduras", "Jamaica", "Mexico", "Montserrat", "Nicaragua", "Panama", "Puerto Rico",
        "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Suriname",
        "Trinidad and Tobago", "Turks and Caicos Islands", "United States", "United States Virgin Islands",
    },
    "CONMEBOL": {
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay", "Venezuela",
    },
    "OFC": {
        "American Samoa", "Cook Islands", "Fiji", "New Caledonia", "New Zealand", "Papua New Guinea",
        "Samoa", "Solomon Islands", "Tahiti", "Tonga", "Vanuatu",
    },
    "UEFA": {
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
        "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia",
        "Denmark", "England", "Estonia", "Faroe Islands", "Finland", "France", "Georgia", "Germany",
        "Gibraltar", "Greece", "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan", "Kosovo",
        "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Montenegro",
        "Netherlands", "North Macedonia", "Northern Ireland", "Norway", "Poland", "Portugal",
        "Republic of Ireland", "Romania", "Russia", "San Marino", "Scotland", "Serbia", "Slovakia",
        "Slovenia", "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "Wales",
    },
}


TEAM_TO_CONFEDERATION = {
    team: confederation
    for confederation, teams in CONFEDERATION_TEAMS.items()
    for team in teams
}


@dataclass(frozen=True)
class StrengthRecord:
    team: str
    confederation: str
    rating: float
    source: str = "default"


@dataclass(frozen=True)
class StrengthCorrection:
    prediction: Prediction
    applied: bool
    home_confederation: str | None = None
    away_confederation: str | None = None
    home_rating: float | None = None
    away_rating: float | None = None
    log_adjustment: float = 0.0
    reason: str = ""


def default_strength_records(teams: Iterable[str]) -> pd.DataFrame:
    records = []
    for team in sorted(set(str(team) for team in teams)):
        confederation = TEAM_TO_CONFEDERATION.get(team)
        if not confederation:
            continue
        rating = TEAM_RATING_OVERRIDES.get(team, CONFEDERATION_BASE_RATING[confederation])
        records.append(
            {
                "team": team,
                "confederation": confederation,
                "rating": float(rating),
                "source": "default",
            }
        )
    return pd.DataFrame(records, columns=["team", "confederation", "rating", "source"])


def normalize_strength_frame(frame: pd.DataFrame, source: str = "csv") -> pd.DataFrame:
    normalized = frame.rename(columns={"Team": "team", "Confederation": "confederation", "Rating": "rating"}).copy()
    required = {"team", "confederation", "rating"}
    missing = required - set(normalized.columns)
    if missing:
        raise ValueError(f"Missing required strength columns: {sorted(missing)}")
    normalized["team"] = normalized["team"].astype(str).str.strip()
    normalized["confederation"] = normalized["confederation"].astype(str).str.strip().str.upper()
    normalized["rating"] = pd.to_numeric(normalized["rating"], errors="coerce")
    normalized["source"] = normalized["source"].astype(str) if "source" in normalized.columns else source
    normalized = normalized.dropna(subset=["team", "confederation", "rating"])
    return normalized[["team", "confederation", "rating", "source"]].drop_duplicates("team", keep="last").reset_index(drop=True)


def load_strength_csv(path: str | Path) -> pd.DataFrame:
    return normalize_strength_frame(pd.read_csv(path), source=str(path))


def combine_strength_frames(defaults: pd.DataFrame, overrides: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [defaults] + [frame for frame in overrides if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["team", "confederation", "rating", "source"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates("team", keep="last").reset_index(drop=True)


def strength_lookup(frame: pd.DataFrame) -> dict[str, StrengthRecord]:
    return {
        str(row.team): StrengthRecord(str(row.team), str(row.confederation), float(row.rating), str(row.source))
        for row in frame.itertuples(index=False)
    }


def apply_cross_confederation_strength_correction(
    prediction: Prediction,
    lookup: dict[str, StrengthRecord],
    enabled: bool = True,
    intensity: float = 0.6,
    cross_confederation_only: bool = True,
    max_log_adjustment: float = 0.22,
    rho: float = 0.0,
) -> StrengthCorrection:
    if not enabled:
        return StrengthCorrection(prediction=prediction, applied=False, reason="disabled")
    home = lookup.get(prediction.home_team)
    away = lookup.get(prediction.away_team)
    if home is None or away is None:
        return StrengthCorrection(prediction=prediction, applied=False, reason="missing_rating")
    if cross_confederation_only and home.confederation == away.confederation:
        return StrengthCorrection(
            prediction=prediction,
            applied=False,
            home_confederation=home.confederation,
            away_confederation=away.confederation,
            home_rating=home.rating,
            away_rating=away.rating,
            reason="same_confederation",
        )

    rating_diff = home.rating - away.rating
    raw_adjustment = (rating_diff / 100.0) * 0.045 * max(0.0, min(1.0, intensity))
    log_adjustment = max(-max_log_adjustment, min(max_log_adjustment, raw_adjustment))
    adjusted = prediction_from_expectations(
        prediction.home_team,
        prediction.away_team,
        prediction.home_goal_expectation * exp(log_adjustment),
        prediction.away_goal_expectation * exp(-log_adjustment),
        rho=rho,
        max_goals=prediction.score_matrix.shape[0] - 1,
    )
    return StrengthCorrection(
        prediction=adjusted,
        applied=True,
        home_confederation=home.confederation,
        away_confederation=away.confederation,
        home_rating=home.rating,
        away_rating=away.rating,
        log_adjustment=log_adjustment,
        reason="applied",
    )
