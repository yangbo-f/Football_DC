from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ODDS_COLUMNS = [
    "event_date",
    "competition",
    "home_team",
    "away_team",
    "market",
    "selection",
    "line",
    "odds_decimal",
    "bookmaker",
    "captured_at",
    "source",
]


@dataclass(frozen=True)
class OddsProviderConfig:
    provider: str
    api_key_env: str
    enabled: bool = False


THE_ODDS_API = OddsProviderConfig(provider="The Odds API", api_key_env="THE_ODDS_API_KEY", enabled=False)


def normalize_odds(frame: pd.DataFrame) -> pd.DataFrame:
    odds = frame.copy()
    for column in ODDS_COLUMNS:
        if column not in odds.columns:
            odds[column] = ""
    odds = odds[ODDS_COLUMNS].copy()
    odds["event_date"] = pd.to_datetime(odds["event_date"], errors="coerce")
    odds["line"] = pd.to_numeric(odds["line"], errors="coerce")
    odds["odds_decimal"] = pd.to_numeric(odds["odds_decimal"], errors="coerce")
    return odds.dropna(subset=["event_date", "competition", "home_team", "away_team", "market", "selection", "odds_decimal"]).reset_index(drop=True)


def load_odds_files(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [normalize_odds(pd.read_csv(path)) for path in paths]
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame(columns=ODDS_COLUMNS)
    return pd.concat(usable, ignore_index=True)


def manual_1x2_odds(home_odds: float, draw_odds: float, away_odds: float) -> dict[str, float]:
    return {"Home": home_odds, "Draw": draw_odds, "Away": away_odds}


def match_odds(
    odds: pd.DataFrame,
    competition: str,
    home_team: str,
    away_team: str,
    event_date,
    market: str,
    line: float | None = None,
) -> dict[str, float]:
    if odds.empty:
        return {}
    date_value = pd.to_datetime(event_date).date()
    filtered = odds[
        (odds["competition"].astype(str) == competition)
        & (odds["home_team"].astype(str) == home_team)
        & (odds["away_team"].astype(str) == away_team)
        & (odds["event_date"].dt.date == date_value)
        & (odds["market"].astype(str) == market)
    ].copy()
    if line is not None:
        filtered = filtered[filtered["line"].fillna(9999).round(4) == round(float(line), 4)]
    if filtered.empty:
        return {}
    return dict(zip(filtered["selection"], filtered["odds_decimal"]))


def fetch_odds_placeholder(*_args, **_kwargs) -> pd.DataFrame:
    raise NotImplementedError("Automatic odds API is reserved for a later step. Configure a provider key before enabling it.")
