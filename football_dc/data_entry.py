from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil

import pandas as pd

from .catalog import DataSource
from .data import MATCH_COLUMNS, OPTIONAL_MATCH_COLUMNS, default_neutral_site, normalize_matches, normalize_team_name


ENTRY_COLUMNS = MATCH_COLUMNS + OPTIONAL_MATCH_COLUMNS


@dataclass(frozen=True)
class MatchEntry:
    competition: str
    season: str
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    neutral_site: bool
    stage: str = ""
    round: str = ""
    score_basis: str = "FT90"
    decided_by_penalties: bool = False
    winner: str = ""
    notes: str = ""

    def as_row(self) -> dict[str, object]:
        return {
            "competition": self.competition,
            "season": self.season,
            "date": self.date,
            "home_team": normalize_team_name(self.home_team),
            "away_team": normalize_team_name(self.away_team),
            "home_goals": int(self.home_goals),
            "away_goals": int(self.away_goals),
            "neutral_site": bool(self.neutral_site),
            "stage": self.stage,
            "round": self.round,
            "score_basis": self.score_basis or "FT90",
            "decided_by_penalties": bool(self.decided_by_penalties),
            "winner": normalize_team_name(self.winner),
            "notes": self.notes,
        }


def default_entry_values(source: DataSource) -> dict[str, object]:
    season = ""
    parts = source.label.split()
    if len(parts) >= 2:
        season = parts[1]
    stage = ""
    round_name = ""
    if source.competition in {"WorldCup", "WomenWorldCup"}:
        stage = "Group"
        round_name = "Group Stage"
    elif source.competition in {"WorldCupQualifiers", "WomenWorldCupQualifiers"}:
        stage = "Qualification"
        round_name = "Qualification"
    elif source.competition == "ChampionsLeague":
        stage = "League"
        round_name = "League Phase"
    elif source.competition == "ChampionsLeagueQualifiers":
        stage = "Qualification"
        round_name = "Qualification"
    elif source.competition in {"EPL", "CSL"}:
        stage = "League"
    return {
        "competition": source.competition,
        "season": season,
        "neutral_site": default_neutral_site(source.competition),
        "score_basis": "FT90",
        "stage": stage,
        "round": round_name,
    }


def target_teams(source: DataSource) -> list[str]:
    if not source.path.exists():
        return []
    try:
        matches = normalize_matches(pd.read_csv(source.path))
    except Exception:
        return []
    teams = set(matches["home_team"].astype(str)) | set(matches["away_team"].astype(str))
    return sorted(team for team in teams if team)


def validate_entry(entry: MatchEntry) -> list[str]:
    errors = []
    if not entry.competition.strip():
        errors.append("缺少赛事。")
    if not entry.season.strip():
        errors.append("缺少赛季。")
    if not entry.home_team.strip() or not entry.away_team.strip():
        errors.append("主队和客队不能为空。")
    if entry.home_team.strip() == entry.away_team.strip():
        errors.append("主队和客队不能相同。")
    if int(entry.home_goals) < 0 or int(entry.away_goals) < 0:
        errors.append("进球数必须是非负整数。")
    parsed_date = pd.to_datetime(entry.date, errors="coerce")
    if pd.isna(parsed_date):
        errors.append("日期格式无效。")
    return errors


def read_target_table(path: Path) -> pd.DataFrame:
    if path.exists():
        raw = pd.read_csv(path)
    else:
        raw = pd.DataFrame(columns=ENTRY_COLUMNS)
    for column in ENTRY_COLUMNS:
        if column not in raw.columns:
            raw[column] = ""
    return raw[ENTRY_COLUMNS].copy()


def is_duplicate(raw: pd.DataFrame, entry: MatchEntry) -> bool:
    if raw.empty:
        return False
    dates = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    entry_date = pd.to_datetime(entry.date).strftime("%Y-%m-%d")
    duplicate = (
        raw["competition"].astype(str).eq(entry.competition)
        & dates.eq(entry_date)
        & raw["home_team"].map(normalize_team_name).astype(str).eq(str(normalize_team_name(entry.home_team)))
        & raw["away_team"].map(normalize_team_name).astype(str).eq(str(normalize_team_name(entry.away_team)))
    )
    return bool(duplicate.any())


def backup_csv(path: Path) -> Path | None:
    if not path.exists():
        return None
    backup_dir = path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.stem}_{timestamp}.csv"
    shutil.copy2(path, backup_path)
    return backup_path


def append_match_entry(source: DataSource, entry: MatchEntry) -> Path | None:
    errors = validate_entry(entry)
    if errors:
        raise ValueError(" ".join(errors))
    raw = read_target_table(source.path)
    if is_duplicate(raw, entry):
        raise ValueError("检测到重复比赛：同一赛事、日期、主队、客队已经存在。")
    backup_path = backup_csv(source.path)
    updated = pd.concat([raw, pd.DataFrame([entry.as_row()])], ignore_index=True)
    updated["date"] = pd.to_datetime(updated["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    source.path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(source.path, index=False, encoding="utf-8")
    return backup_path
