from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data import combine_match_frames, load_matches


@dataclass(frozen=True)
class DataSource:
    key: str
    label: str
    competition: str
    path: Path
    default: bool = False


SOURCE_GROUPS = {
    "WorldCup": "世界杯",
    "WorldCupQualifiers": "世界杯",
    "WomenWorldCup": "女足世界杯",
    "WomenWorldCupQualifiers": "女足世界杯",
    "ChampionsLeague": "欧冠",
    "ChampionsLeagueQualifiers": "欧冠",
    "EPL": "英超",
    "CSL": "中超",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIRECTORIES = {
    "worldcup": PROJECT_ROOT / "data/worldcup",
    "women_worldcup": PROJECT_ROOT / "data/women_worldcup",
    "champions_league": PROJECT_ROOT / "data/champions_league",
    "epl": PROJECT_ROOT / "data/epl",
    "csl": PROJECT_ROOT / "data/csl",
}


def source_by_label() -> dict[str, DataSource]:
    return {source.label: source for source in discover_data_sources()}


def default_source_labels() -> list[str]:
    return [source.label for source in discover_data_sources() if source.default]


def source_group(source: DataSource) -> str:
    return SOURCE_GROUPS.get(source.competition, source.competition)


def source_short_label(source: DataSource) -> str:
    for prefix in ("世界杯 ", "女足世界杯 ", "欧冠 ", "英超 ", "中超 "):
        if source.label.startswith(prefix):
            return source.label.removeprefix(prefix)
    return source.label


def source_match_count(source: DataSource) -> int:
    return cached_source_match_count(str(source.path))


@lru_cache(maxsize=64)
def cached_source_match_count(path: str) -> int:
    source_path = Path(path)
    if not source_path.exists():
        return 0
    try:
        return len(load_matches(source_path))
    except Exception:
        return 0


def clear_catalog_cache() -> None:
    cached_source_match_count.cache_clear()


def source_display_label(source: DataSource) -> str:
    return f"{source_short_label(source)} · {source_match_count(source)}场"


def discover_data_sources() -> list[DataSource]:
    sources = []
    for directory in DATA_DIRECTORIES.values():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            source = source_from_path(path)
            if source is not None and source_match_count(source) > 0:
                sources.append(source)
    return sorted(sources, key=source_sort_key)


def source_from_path(path: Path) -> DataSource | None:
    name = path.stem
    parent = path.parent.name
    if parent == "worldcup":
        if name.startswith("finals_"):
            season = name.removeprefix("finals_")
            return DataSource(f"worldcup_{name}", f"世界杯 {season} 正赛", "WorldCup", path, default=(season == "2026"))
        if name.startswith("qualifiers_") and name.endswith("_cycle_all"):
            season = name.removeprefix("qualifiers_").removesuffix("_cycle_all")
            return DataSource(f"worldcup_{name}", f"世界杯 {season} 预选赛周期", "WorldCupQualifiers", path)
        return None
    if parent == "women_worldcup":
        if name.startswith("finals_"):
            season = name.removeprefix("finals_")
            return DataSource(f"women_worldcup_{name}", f"女足世界杯 {season} 正赛", "WomenWorldCup", path)
        if name.startswith("qualifiers_") and name.endswith("_cycle_all"):
            season = name.removeprefix("qualifiers_").removesuffix("_cycle_all")
            return DataSource(f"women_worldcup_{name}", f"女足世界杯 {season} 预选赛周期", "WomenWorldCupQualifiers", path)
        return None
    if parent == "champions_league":
        if name.startswith("main_"):
            season = name.removeprefix("main_").replace("_", "-")
            return DataSource(f"champions_league_{name}", f"欧冠 {season} 正赛", "ChampionsLeague", path)
        if name.startswith("qualifiers_"):
            season = name.removeprefix("qualifiers_").replace("_", "-")
            return DataSource(f"champions_league_{name}", f"欧冠 {season} 预选赛", "ChampionsLeagueQualifiers", path)
        return None
    if parent == "epl" and name.startswith("epl_"):
        season = name.removeprefix("epl_").replace("_", "-")
        return DataSource(name, f"英超 {season}", "EPL", path)
    if parent == "csl" and name.startswith("csl_"):
        season = name.removeprefix("csl_").replace("_", "-")
        return DataSource(name, f"中超 {season}", "CSL", path)
    return None


def source_sort_key(source: DataSource) -> tuple:
    group_rank = {
        "WorldCup": 0,
        "WorldCupQualifiers": 0,
        "WomenWorldCup": 1,
        "WomenWorldCupQualifiers": 1,
        "ChampionsLeague": 2,
        "ChampionsLeagueQualifiers": 2,
        "EPL": 3,
        "CSL": 4,
    }.get(source.competition, 9)
    qualifier_rank = 1 if source.competition in {"WorldCupQualifiers", "WomenWorldCupQualifiers", "ChampionsLeagueQualifiers"} else 0
    return (group_rank, source_group(source), source.label, qualifier_rank)


def load_catalog_sources(labels: Iterable[str]) -> tuple[pd.DataFrame, list[str]]:
    label_map = source_by_label()
    frames = []
    loaded = []
    for label in labels:
        source = label_map[label]
        if not source.path.exists():
            continue
        frame = load_matches(source.path)
        if frame.empty:
            continue
        frame["source_file"] = str(source.path.relative_to(PROJECT_ROOT))
        frames.append(frame)
        loaded.append(frame["source_file"].iloc[0])
    return combine_match_frames(frames), loaded
