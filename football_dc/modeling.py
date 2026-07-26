from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import filter_competition, filter_training_matches
from .dixon_coles import DixonColesModel


@dataclass(frozen=True)
class TrainedCompetitionModel:
    competition: str
    model: DixonColesModel
    matches_used: int
    first_match_date: pd.Timestamp
    last_match_date: pd.Timestamp
    teams_used: int
    excluded_non_ft90: int
    weighted_matches: float


def train_competition_model(matches: pd.DataFrame, competition: str, half_life_days: float = 365.0) -> TrainedCompetitionModel:
    competition_rows = filter_competition(matches, competition)
    filtered = filter_training_matches(matches, competition)
    if filtered.empty:
        raise ValueError(f"No FT90 matches available for competition: {competition}")
    filtered = apply_training_match_weights(filtered, competition)
    model = DixonColesModel(half_life_days=half_life_days).fit(filtered)
    return TrainedCompetitionModel(
        competition=competition,
        model=model,
        matches_used=len(filtered),
        first_match_date=filtered["date"].min(),
        last_match_date=filtered["date"].max(),
        teams_used=len(set(filtered["home_team"]) | set(filtered["away_team"])),
        excluded_non_ft90=max(0, len(competition_rows) - len(filtered)),
        weighted_matches=float(filtered["match_weight"].sum()) if "match_weight" in filtered.columns else float(len(filtered)),
    )


def apply_training_match_weights(matches: pd.DataFrame, competition: str) -> pd.DataFrame:
    weighted = matches.copy()
    weighted["match_weight"] = 1.0
    qualifier_codes = {
        "WorldCup": "WorldCupQualifiers",
        "WomenWorldCup": "WomenWorldCupQualifiers",
    }
    qualifier_code = qualifier_codes.get(competition)
    if qualifier_code and "competition" in weighted.columns:
        is_qualifier = weighted["competition"].astype(str) == qualifier_code
        weighted.loc[is_qualifier, "match_weight"] = 0.45
        if "stage" in weighted.columns:
            stage = weighted["stage"].fillna("").astype(str).str.lower()
            knockout = (weighted["competition"].astype(str) == competition) & stage.str.contains("knockout")
            weighted.loc[knockout, "match_weight"] = 1.05
    return weighted
