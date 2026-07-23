from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import STANDARD_MATCH_COLUMNS, load_matches


MATCH_KEY_COLUMNS = ["competition", "date", "home_team", "away_team"]


@dataclass(frozen=True)
class MatchDataQualityReport:
    source: str
    total_rows: int
    missing_columns: list[str]
    duplicate_rows: int
    missing_score_rows: int
    invalid_goal_rows: int
    same_team_rows: int
    non_ft90_rows: int
    missing_match_importance_rows: int
    future_prediction_rows: int

    @property
    def has_errors(self) -> bool:
        return bool(
            self.missing_columns
            or self.duplicate_rows
            or self.missing_score_rows
            or self.invalid_goal_rows
            or self.same_team_rows
            or self.future_prediction_rows
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "total_rows": self.total_rows,
            "missing_columns": ", ".join(self.missing_columns),
            "duplicate_rows": self.duplicate_rows,
            "missing_score_rows": self.missing_score_rows,
            "invalid_goal_rows": self.invalid_goal_rows,
            "same_team_rows": self.same_team_rows,
            "non_ft90_rows": self.non_ft90_rows,
            "missing_match_importance_rows": self.missing_match_importance_rows,
            "future_prediction_rows": self.future_prediction_rows,
            "has_errors": self.has_errors,
        }


def check_match_data(matches: pd.DataFrame, source: str = "dataframe") -> MatchDataQualityReport:
    missing_columns = [column for column in STANDARD_MATCH_COLUMNS if column not in matches.columns]
    if missing_columns:
        return MatchDataQualityReport(
            source=source,
            total_rows=len(matches),
            missing_columns=missing_columns,
            duplicate_rows=0,
            missing_score_rows=0,
            invalid_goal_rows=0,
            same_team_rows=0,
            non_ft90_rows=0,
            missing_match_importance_rows=0,
            future_prediction_rows=0,
        )

    checked = matches.copy()
    checked["date"] = pd.to_datetime(checked["date"], errors="coerce")
    checked["prediction_available_at"] = pd.to_datetime(checked["prediction_available_at"], errors="coerce")
    home_goals = pd.to_numeric(checked["home_goals"], errors="coerce")
    away_goals = pd.to_numeric(checked["away_goals"], errors="coerce")
    importance = pd.to_numeric(checked["match_importance"], errors="coerce")

    duplicate_rows = int(checked.duplicated(MATCH_KEY_COLUMNS, keep=False).sum())
    missing_score_rows = int((home_goals.isna() | away_goals.isna()).sum())
    invalid_goal_rows = int(((home_goals < 0) | (away_goals < 0)).fillna(False).sum())
    same_team_rows = int((checked["home_team"].astype(str) == checked["away_team"].astype(str)).sum())
    score_basis = checked["score_basis"].fillna("FT90").replace("", "FT90").astype(str).str.upper()
    non_ft90_rows = int((score_basis != "FT90").sum())
    missing_match_importance_rows = int(importance.isna().sum())
    future_prediction_rows = int((checked["prediction_available_at"] > checked["date"]).fillna(False).sum())

    return MatchDataQualityReport(
        source=source,
        total_rows=len(checked),
        missing_columns=[],
        duplicate_rows=duplicate_rows,
        missing_score_rows=missing_score_rows,
        invalid_goal_rows=invalid_goal_rows,
        same_team_rows=same_team_rows,
        non_ft90_rows=non_ft90_rows,
        missing_match_importance_rows=missing_match_importance_rows,
        future_prediction_rows=future_prediction_rows,
    )


def check_match_file(path: str | Path) -> MatchDataQualityReport:
    source_path = Path(path)
    matches = load_matches(source_path)
    return check_match_data(matches, source=str(source_path))


def reports_to_frame(reports: list[MatchDataQualityReport]) -> pd.DataFrame:
    return pd.DataFrame([report.to_dict() for report in reports])
