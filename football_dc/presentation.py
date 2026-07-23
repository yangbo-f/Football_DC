from __future__ import annotations

import pandas as pd

from .team_names import team_display_name


SELECTION_LABELS = {
    "Home": "主胜",
    "Draw": "平局",
    "Away": "客胜",
    "Over": "大",
    "Under": "小",
    "Yes": "是",
    "No": "否",
    "Push": "走盘",
}


def selection_label(selection: str) -> str:
    return SELECTION_LABELS.get(selection, selection)


def matchup_display(home_team: str, away_team: str) -> str:
    return f"{team_display_name(home_team)} vs {team_display_name(away_team)}"


def probability_cell_style(value: float) -> str:
    if not isinstance(value, (float, int)):
        return ""
    clamped = max(0.0, min(float(value), 0.25))
    intensity = clamped / 0.25
    green = int(248 - 116 * intensity)
    blue = int(244 - 118 * intensity)
    return f"background-color: rgb(232, {green}, {blue}); color: #123524; font-weight: 600;"


def styled_probability_matrix(frame: pd.DataFrame):
    return frame.style.format(lambda value: f"{value * 100:.1f}%").map(probability_cell_style)
