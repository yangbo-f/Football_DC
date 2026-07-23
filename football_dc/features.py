from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log
from statistics import mean, pstdev
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 24.0
    home_advantage: float = 60.0
    form_half_life_days: float = 365.0


XG_COLUMN_ALIASES = {
    "xg_home": "home_xg",
    "home_xg": "home_xg",
    "Home xG": "home_xg",
    "home_xg_for": "home_xg",
    "xg_away": "away_xg",
    "away_xg": "away_xg",
    "Away xG": "away_xg",
    "away_xg_for": "away_xg",
}


def build_pre_match_features(matches: pd.DataFrame, config: EloConfig | None = None) -> pd.DataFrame:
    """Generate strictly pre-match Elo, form, schedule strength, and optional xG features."""
    cfg = config or EloConfig()
    prepared = _prepare_feature_matches(matches)
    ratings: dict[str, float] = defaultdict(lambda: cfg.base_rating)
    team_history: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rating_history: dict[str, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for row in prepared.itertuples(index=False):
        current = row._asdict()
        match_date = pd.Timestamp(current["date"])
        home_team = str(current["home_team"])
        away_team = str(current["away_team"])
        home_elo = float(ratings[home_team])
        away_elo = float(ratings[away_team])

        feature_row = dict(current)
        feature_row.update(_elo_feature_block("home", home_team, match_date, home_elo, rating_history, cfg))
        feature_row.update(_elo_feature_block("away", away_team, match_date, away_elo, rating_history, cfg))
        feature_row["elo_diff"] = home_elo - away_elo
        feature_row.update(_form_feature_block("home", team_history[home_team], match_date, cfg))
        feature_row.update(_form_feature_block("away", team_history[away_team], match_date, cfg))
        feature_row.update(_diff_features(feature_row))
        rows.append(feature_row)

        home_goals = int(current["home_goals"])
        away_goals = int(current["away_goals"])
        home_points, away_points = _result_points(home_goals, away_goals)
        match_importance = float(current.get("match_importance", 1.0) or 1.0)
        home_expected = _expected_score(home_elo, away_elo, False if current.get("neutral_site") else True, cfg)
        goal_multiplier = _goal_difference_multiplier(abs(home_goals - away_goals))
        elo_delta = cfg.k_factor * match_importance * goal_multiplier * (home_points - home_expected)
        ratings[home_team] = home_elo + elo_delta
        ratings[away_team] = away_elo - elo_delta
        rating_history[home_team].append((match_date, float(ratings[home_team])))
        rating_history[away_team].append((match_date, float(ratings[away_team])))

        home_xg = current.get("home_xg")
        away_xg = current.get("away_xg")
        team_history[home_team].append(
            _history_record(match_date, home_points, away_elo, match_importance, home_xg, away_xg)
        )
        team_history[away_team].append(
            _history_record(match_date, away_points, home_elo, match_importance, away_xg, home_xg)
        )

    return pd.DataFrame(rows)


def _prepare_feature_matches(matches: pd.DataFrame) -> pd.DataFrame:
    prepared = matches.rename(columns={column: XG_COLUMN_ALIASES[column] for column in matches.columns if column in XG_COLUMN_ALIASES}).copy()
    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
    prepared["home_goals"] = pd.to_numeric(prepared["home_goals"], errors="coerce")
    prepared["away_goals"] = pd.to_numeric(prepared["away_goals"], errors="coerce")
    if "neutral_site" not in prepared.columns:
        prepared["neutral_site"] = False
    if "match_importance" not in prepared.columns:
        prepared["match_importance"] = 1.0
    if "home_xg" not in prepared.columns:
        prepared["home_xg"] = pd.NA
    if "away_xg" not in prepared.columns:
        prepared["away_xg"] = pd.NA
    prepared["home_xg"] = pd.to_numeric(prepared["home_xg"], errors="coerce")
    prepared["away_xg"] = pd.to_numeric(prepared["away_xg"], errors="coerce")
    prepared = prepared.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    prepared["_feature_order"] = range(len(prepared))
    return prepared.sort_values(["date", "_feature_order"]).drop(columns=["_feature_order"]).reset_index(drop=True)


def _elo_feature_block(prefix: str, team: str, match_date: pd.Timestamp, current_elo: float, history: dict[str, list[tuple[pd.Timestamp, float]]], cfg: EloConfig) -> dict[str, float]:
    recent = [rating for date, rating in history[team] if (match_date - date).days <= 365]
    return {
        f"{prefix}_elo": current_elo,
        f"{prefix}_elo_30d_change": current_elo - _rating_as_of(history[team], match_date - pd.Timedelta(days=30), cfg.base_rating),
        f"{prefix}_elo_90d_change": current_elo - _rating_as_of(history[team], match_date - pd.Timedelta(days=90), cfg.base_rating),
        f"{prefix}_elo_180d_change": current_elo - _rating_as_of(history[team], match_date - pd.Timedelta(days=180), cfg.base_rating),
        f"{prefix}_elo_peak_1y": max(recent) if recent else current_elo,
        f"{prefix}_elo_std_1y": pstdev(recent) if len(recent) > 1 else 0.0,
    }


def _rating_as_of(history: list[tuple[pd.Timestamp, float]], cutoff: pd.Timestamp, fallback: float) -> float:
    eligible = [rating for date, rating in history if date <= cutoff]
    return float(eligible[-1]) if eligible else fallback


def _form_feature_block(prefix: str, history: list[dict[str, Any]], match_date: pd.Timestamp, cfg: EloConfig) -> dict[str, float]:
    return {
        f"{prefix}_form_3": _mean_points(history[-3:]),
        f"{prefix}_form_5": _mean_points(history[-5:]),
        f"{prefix}_form_10": _mean_points(history[-10:]),
        f"{prefix}_weighted_form_5": _weighted_form(history[-5:], match_date, cfg),
        f"{prefix}_weighted_form_10": _weighted_form(history[-10:], match_date, cfg),
        f"{prefix}_avg_opponent_elo_5": _mean_value(history[-5:], "opponent_elo"),
        f"{prefix}_avg_opponent_elo_10": _mean_value(history[-10:], "opponent_elo"),
        f"{prefix}_strength_of_schedule": _mean_value(history[-10:], "opponent_elo") - cfg.base_rating if history else 0.0,
        f"{prefix}_rolling_xG_3": _rolling_xg(history[-3:], "xg_for"),
        f"{prefix}_rolling_xG_5": _rolling_xg(history[-5:], "xg_for"),
        f"{prefix}_rolling_xG_10": _rolling_xg(history[-10:], "xg_for"),
        f"{prefix}_rolling_xGA_3": _rolling_xg(history[-3:], "xg_against"),
        f"{prefix}_rolling_xGA_5": _rolling_xg(history[-5:], "xg_against"),
        f"{prefix}_rolling_xGA_10": _rolling_xg(history[-10:], "xg_against"),
        f"{prefix}_rolling_xGD_5": _rolling_xgd(history[-5:]),
        f"{prefix}_rolling_xGD_10": _rolling_xgd(history[-10:]),
    }


def _diff_features(row: dict[str, Any]) -> dict[str, float]:
    pairs = [
        "form_3",
        "form_5",
        "form_10",
        "weighted_form_5",
        "weighted_form_10",
        "avg_opponent_elo_5",
        "avg_opponent_elo_10",
        "strength_of_schedule",
        "rolling_xG_5",
        "rolling_xGA_5",
        "rolling_xGD_5",
    ]
    return {f"{name}_diff": _safe_diff(row.get(f"home_{name}"), row.get(f"away_{name}")) for name in pairs}


def _safe_diff(left, right) -> float:
    if pd.isna(left) or pd.isna(right):
        return pd.NA
    return float(left) - float(right)


def _history_record(match_date: pd.Timestamp, points: float, opponent_elo: float, importance: float, xg_for, xg_against) -> dict[str, Any]:
    return {
        "date": match_date,
        "points": points,
        "opponent_elo": float(opponent_elo),
        "importance": float(importance),
        "xg_for": xg_for,
        "xg_against": xg_against,
    }


def _mean_points(records: list[dict[str, Any]]) -> float:
    return _mean_value(records, "points")


def _mean_value(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records if not pd.isna(record.get(key))]
    return float(mean(values)) if values else 0.0


def _weighted_form(records: list[dict[str, Any]], match_date: pd.Timestamp, cfg: EloConfig) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for record in records:
        age_days = max(0, (match_date - record["date"]).days)
        time_weight = exp(-log(2.0) * age_days / cfg.form_half_life_days)
        opponent_factor = float(record["opponent_elo"]) / cfg.base_rating
        weight = time_weight * float(record["importance"]) * opponent_factor
        weighted_sum += float(record["points"]) * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight else 0.0


def _rolling_xg(records: list[dict[str, Any]], key: str):
    values = [float(record[key]) for record in records if not pd.isna(record.get(key))]
    if not values:
        return pd.NA
    return float(mean(values))


def _rolling_xgd(records: list[dict[str, Any]]):
    xg_for = _rolling_xg(records, "xg_for")
    xg_against = _rolling_xg(records, "xg_against")
    if pd.isna(xg_for) or pd.isna(xg_against):
        return pd.NA
    return float(xg_for) - float(xg_against)


def _result_points(home_goals: int, away_goals: int) -> tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def _expected_score(home_elo: float, away_elo: float, has_home_advantage: bool, cfg: EloConfig) -> float:
    adjusted_home = home_elo + (cfg.home_advantage if has_home_advantage else 0.0)
    return 1.0 / (1.0 + 10.0 ** ((away_elo - adjusted_home) / 400.0))


def _goal_difference_multiplier(goal_difference: int) -> float:
    if goal_difference <= 1:
        return 1.0
    return 1.0 + log(float(goal_difference))
