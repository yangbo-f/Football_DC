from __future__ import annotations

import base64
from datetime import date
from io import StringIO
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_dc.adjustments import ManualAdjustments, apply_manual_adjustments
from football_dc.catalog import clear_catalog_cache, default_source_labels, discover_data_sources, load_catalog_sources, source_by_label, source_display_label, source_group, source_match_count
from football_dc.data import COMPETITIONS, combine_match_frames, filter_competition, normalize_matches, teams_for_competition
from football_dc.data_entry import MatchEntry, append_match_entry, default_entry_values, target_teams
from football_dc.forecast import assess_match_risk, model_scope_summary
from football_dc.markets import (
    asian_handicap_market,
    btts_market,
    european_handicap_market,
    half_full_time_market,
    over_under_lines,
    result_market,
    total_goals_distribution,
    totals_market,
    value_rows_with_blend,
    remove_vig,
)
from football_dc.modeling import train_competition_model
from football_dc.odds import match_odds, normalize_odds
from football_dc.presentation import matchup_display, selection_label, styled_probability_matrix
from football_dc.strength import (
    apply_cross_confederation_strength_correction,
    combine_strength_frames,
    default_strength_records,
    load_strength_csv,
    normalize_strength_frame,
    strength_lookup,
)
from football_dc.team_names import all_team_names_zh, save_custom_team_name, team_display_name


NO_TEAM_SELECTION = "__NO_TEAM_SELECTION__"
DEFAULT_STRENGTH_RATINGS_PATH = PROJECT_ROOT / "data/worldcup/team_strength_ratings.csv"
LOCAL_STATE_PATH = PROJECT_ROOT / ".streamlit/local_state.json"
LOGO_ASSET_PATH = PROJECT_ROOT / "assets/logo.png"


def pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def decimal(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}"


def parse_decimal_input(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def load_uploaded(uploaded_files) -> list[pd.DataFrame]:
    frames = []
    for uploaded in uploaded_files:
        frames.append(normalize_matches(pd.read_csv(uploaded)))
    return frames


def load_uploaded_odds(uploaded_files) -> pd.DataFrame:
    frames = []
    for uploaded in uploaded_files:
        frames.append(normalize_odds(pd.read_csv(uploaded)))
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame()
    return pd.concat(usable, ignore_index=True)


def load_uploaded_strength(uploaded_files) -> list[pd.DataFrame]:
    frames = []
    for uploaded in uploaded_files:
        frames.append(normalize_strength_frame(pd.read_csv(uploaded), source=getattr(uploaded, "name", "uploaded")))
    return frames


def load_project_strength_frames() -> list[pd.DataFrame]:
    if not DEFAULT_STRENGTH_RATINGS_PATH.exists():
        return []
    return [load_strength_csv(DEFAULT_STRENGTH_RATINGS_PATH)]


def matches_cache_payload(matches: pd.DataFrame) -> str:
    stable = matches.sort_values([column for column in ["competition", "date", "home_team", "away_team"] if column in matches.columns]).reset_index(drop=True)
    return stable.to_json(date_format="iso", orient="split")


@st.cache_resource(show_spinner=False)
def cached_train_model(matches_payload: str, competition: str, half_life_days: float):
    matches = pd.read_json(StringIO(matches_payload), orient="split")
    if "date" in matches.columns:
        matches["date"] = pd.to_datetime(matches["date"])
    return train_competition_model(matches, competition, half_life_days=half_life_days)


def optimize_worldcup_qualifiers(matches: pd.DataFrame, competition: str, home_team: str, away_team: str, enabled: bool) -> pd.DataFrame:
    if not enabled or competition != "WorldCup" or "competition" not in matches.columns:
        return matches
    is_qualifier = matches["competition"].astype(str) == "WorldCupQualifiers"
    if not is_qualifier.any():
        return matches
    focus_teams = {home_team, away_team}
    related_qualifier = is_qualifier & (matches["home_team"].isin(focus_teams) | matches["away_team"].isin(focus_teams))
    return matches[(~is_qualifier) | related_qualifier].copy().reset_index(drop=True)


def optimize_training_matches(matches: pd.DataFrame, competition: str, home_team: str, away_team: str, enabled: bool) -> pd.DataFrame:
    if not enabled or "competition" not in matches.columns:
        return matches
    optimized = optimize_worldcup_qualifiers(matches, competition, home_team, away_team, enabled)
    if competition not in {"EPL", "CSL"}:
        return optimized

    competition_rows = optimized[optimized["competition"].astype(str) == competition].copy()
    league_fast_limit = 800
    if len(competition_rows) <= league_fast_limit:
        return optimized

    recent_indexes = set(competition_rows.sort_values("date").tail(league_fast_limit).index)
    is_selected_competition = optimized["competition"].astype(str) == competition
    keep_rows = (~is_selected_competition) | optimized.index.isin(recent_indexes)
    return optimized[keep_rows].copy().reset_index(drop=True)


def market_table(rows) -> pd.DataFrame:
    frame = pd.DataFrame([row.to_dict() for row in rows])
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "market": "市场",
            "selection": "选项",
            "line": "盘口",
            "model_probability": "模型概率",
            "odds_decimal": "赔率",
            "fair_odds": "公平赔率",
            "implied_probability": "隐含概率",
            "no_vig_probability": "去水概率",
            "edge": "概率差",
            "expected_value": "EV",
        }
    )
    frame["选项"] = frame["选项"].map(selection_label)
    return frame


def blended_market_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "market": "市场",
            "selection": "选项",
            "line": "盘口",
            "model_probability": "纯模型概率",
            "blended_probability": "融合概率",
            "odds_decimal": "赔率",
            "no_vig_probability": "去水概率（No-vig）",
            "model_ev": "纯模型期望收益（EV）",
            "blended_ev": "融合期望收益（EV）",
        }
    )
    frame["选项"] = frame["选项"].map(selection_label)
    return frame


def format_market_table(frame: pd.DataFrame):
    return frame.style.format(
        {
            "模型概率": pct,
            "隐含概率": pct,
            "去水概率": pct,
            "概率差": pct,
            "EV": pct,
            "赔率": decimal,
            "公平赔率": decimal,
        }
    ).map(value_style, subset=["EV", "概率差"])


def format_blended_table(frame: pd.DataFrame):
    return frame.style.format(
        {
            "纯模型概率": pct,
            "融合概率": pct,
            "去水概率（No-vig）": pct,
            "赔率": decimal,
            "纯模型期望收益（EV）": pct,
            "融合期望收益（EV）": pct,
        }
    ).map(value_style, subset=["纯模型期望收益（EV）", "融合期望收益（EV）"])


def filter_value_table(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if mode == "正 EV":
        return frame[(frame["纯模型期望收益（EV）"] > 0) | (frame["融合期望收益（EV）"] > 0)].reset_index(drop=True)
    if mode == "已填赔率":
        return frame[frame["赔率"].notna() & (frame["赔率"] > 0)].reset_index(drop=True)
    return frame


def value_style(value):
    if not isinstance(value, float):
        return "color: #8a8f98;"
    if value > 0:
        return "color: #11823b; font-weight: 700"
    if value < 0:
        return "color: #b42318;"
    return "color: #666;"


def inject_styles() -> None:
    logo_uri = logo_data_uri()
    sidebar_logo_value = f'url("{logo_uri}")' if logo_uri else "none"
    st.markdown(
        f"""
        <style>
        :root {{
            --sidebar-logo: {sidebar_logo_value};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f5f7fb;
            --card-bg: #ffffff;
            --line: #dbe5f3;
            --line-soft: #edf2f7;
            --text: #0f172a;
            --muted: #64748b;
            --blue: #2563eb;
            --green: #16a34a;
            --amber: #d97706;
            --red: #dc2626;
        }
        html, body, .stApp, .block-container, [data-testid="stAppViewContainer"],
        [data-testid="stMarkdownContainer"], label, input, textarea, button {
            font-size: 14px;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 28rem),
                var(--app-bg);
        }
        div[data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 28rem),
                var(--app-bg);
        }
        .block-container {
            padding-top: 1.55rem;
            padding-left: 1.75rem;
            padding-right: 1.75rem;
            max-width: 1118px;
        }
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid var(--line);
            box-shadow: none;
            width: 256px;
            min-width: 256px;
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 0.8rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
            width: 256px;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 56px !important;
            min-width: 56px !important;
            transform: none !important;
            visibility: visible !important;
            overflow: hidden;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] > div {
            width: 56px !important;
            min-width: 56px !important;
            padding: 0.75rem 0.45rem !important;
            overflow: hidden;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] > div::before {
            content: "";
            display: block;
            width: 38px;
            height: 38px;
            margin: 2.9rem auto 0.65rem auto;
            border-radius: 10px;
            background-image: var(--sidebar-logo);
            background-size: cover;
            background-position: center;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }
        section[data-testid="stSidebar"][aria-expanded="false"] > div::after {
            content: "";
            display: block;
            margin: 0;
        }
        .sidebar-collapsed-summary {
            display: none;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] .sidebar-collapsed-summary {
            display: block;
            margin: 0.25rem auto 0 auto;
            color: #64748b;
            font-size: 0.75rem;
            font-weight: 800;
            writing-mode: vertical-rl;
            letter-spacing: 0.08rem;
            max-height: 220px;
            overflow: hidden;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] .sidebar-brand,
        section[data-testid="stSidebar"][aria-expanded="false"] .sidebar-section-label,
        section[data-testid="stSidebar"][aria-expanded="false"] .stRadio,
        section[data-testid="stSidebar"][aria-expanded="false"] .stButton,
        section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stExpander"],
        section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stFileUploader"],
        section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stNumberInput"],
        section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stCheckbox"],
        section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSlider"] {
            display: none !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] ~ div[data-testid="stAppViewContainer"] .block-container {
            max-width: calc(100vw - 96px);
            padding-left: 1.75rem;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.5rem 0;
            border-color: #e2e8f0;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            color: var(--text);
            letter-spacing: 0;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 8px;
            padding: 7px 9px;
            min-height: 34px;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: #eff6ff;
            color: var(--blue);
            font-weight: 700;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
            border: 1px dashed #cbd5e1;
            border-radius: 8px;
            background: #f8fafc;
            padding: 8px;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.25rem 0 0.75rem 0;
            border-bottom: 1px solid #eef2f7;
            margin-bottom: 0.65rem;
        }
        .sidebar-logo {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            object-fit: cover;
            flex: 0 0 auto;
            background: #eff6ff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }
        .sidebar-brand-text {
            min-width: 0;
        }
        .sidebar-brand-title {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 850;
            line-height: 1.25;
        }
        .sidebar-brand-subtitle {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.18rem;
        }
        .sidebar-section-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            margin: 0.75rem 0 0.2rem 0;
        }
        div[data-testid="stExpander"] {
            border: 1px solid var(--line) !important;
            border-radius: 8px !important;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] {
            border-color: #e2e8f0 !important;
            background: #ffffff;
            box-shadow: none;
            margin-bottom: 0.35rem;
            overflow: hidden;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details > summary {
            min-height: 38px;
            padding: 0 10px;
            color: #334155;
            font-size: 0.92rem;
            font-weight: 750;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details[open] > summary {
            border-bottom: 1px solid #e2e8f0;
            background: #ffffff;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] div[data-testid="stVerticalBlock"] {
            gap: 0.25rem;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] label {
            min-height: 30px;
            padding: 4px 0;
        }
        section[data-testid="stSidebar"] .stButton button {
            width: 100%;
            min-height: 34px;
            background: #ffffff;
            border-color: #dbe5f3;
            color: #334155;
            justify-content: center;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background: #eff6ff;
            border-color: #bfdbfe;
            color: #1d4ed8;
        }
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
            color: #475569;
            font-size: 0.86rem;
            font-weight: 700;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            padding: 0.3rem 0.4rem;
        }
        div[data-testid="stSelectbox"] > div,
        div[data-testid="stDateInput"] > div,
        div[data-testid="stNumberInput"] > div,
        div[data-testid="stTextInput"] > div {
            border-radius: 8px;
        }
        .stButton button,
        .stFormSubmitButton button {
            border-radius: 8px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1d4ed8;
            font-weight: 700;
        }
        .stButton button:hover,
        .stFormSubmitButton button:hover {
            border-color: #2563eb;
            color: #1d4ed8;
        }
        .app-hero {
            padding: 0.35rem 0 0.95rem 0;
        }
        .app-title {
            color: #0f172a;
            font-size: 2.55rem;
            line-height: 1.08;
            font-weight: 850;
            margin: 0;
            letter-spacing: 0;
        }
        .app-subtitle {
            color: #475569;
            font-size: 1.05rem;
            margin-top: 0.35rem;
        }
        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 1rem;
        }
        .hero-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid #cfe0ff;
            background: #eef5ff;
            color: #1d4ed8;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 0.9rem;
            font-weight: 700;
        }
        .hero-chip.green {
            border-color: #bbf7d0;
            background: #ecfdf3;
            color: #15803d;
        }
        .hero-chip.gray {
            border-color: #e2e8f0;
            background: #f8fafc;
            color: #475569;
        }
        .panel-card, .summary-strip, .prediction-grid {
            background: var(--card-bg);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            margin: 0.75rem 0 1rem 0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }
        .match-summary {
            border: 1px solid var(--line);
            background: #ffffff;
            border-radius: 8px;
            padding: 20px 22px;
            margin: 1rem 0 1rem 0;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
        }
        .match-summary h3 {
            margin: 0 0 0.25rem 0;
            color: var(--text);
            font-size: 1.7rem;
            line-height: 1.18;
        }
        .muted { color: var(--muted); font-size: 0.95rem; }
        .summary-strip {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
        }
        .summary-item, .prediction-card {
            background: #ffffff;
            border: 1px solid var(--line-soft);
            border-radius: 8px;
            padding: 13px 14px;
            min-height: 86px;
        }
        .summary-label, .prediction-label {
            color: var(--muted);
            font-size: 0.9rem;
            margin-bottom: 4px;
            font-weight: 700;
        }
        .summary-value {
            color: var(--text);
            font-size: 1.05rem;
            font-weight: 650;
        }
        .prediction-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
        }
        .prediction-value {
            color: var(--text);
            font-size: 1.7rem;
            font-weight: 760;
            line-height: 1.2;
        }
        .prediction-card.top-score .prediction-value {
            font-size: 1.25rem;
            line-height: 1.35;
        }
        .top-score-list {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .top-score-item {
            display: flex;
            align-items: baseline;
            gap: 7px;
            white-space: nowrap;
        }
        .top-score-score {
            color: #2563eb;
            font-size: 1.25rem;
            font-weight: 780;
        }
        .top-score-prob {
            color: #94a3b8;
            font-size: 0.86rem;
            font-weight: 650;
        }
        .prediction-card.primary {
            border-color: #bfdbfe;
            background: #f8fbff;
        }
        .prediction-card.good {
            border-color: #bbf7d0;
            background: #f0fdf4;
        }
        .prediction-card.warn {
            border-color: #fed7aa;
            background: #fff7ed;
        }
        .prediction-card.danger {
            border-color: #fecaca;
            background: #fef2f2;
        }
        .match-summary .prediction-grid {
            border: 0;
            background: transparent;
            padding: 0;
            margin: 0.85rem 0 0 0;
        }
        .risk-panel {
            display: grid;
            grid-template-columns: 0.8fr 1fr 1fr;
            gap: 10px;
            margin-top: 12px;
        }
        .risk-card {
            border: 1px solid var(--line-soft);
            border-radius: 8px;
            padding: 12px 14px;
            background: #f8fafc;
        }
        .risk-label {
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .risk-value {
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 760;
        }
        .risk-card.good {
            border-color: #bbf7d0;
            background: #f0fdf4;
        }
        .risk-card.warn {
            border-color: #fed7aa;
            background: #fff7ed;
        }
        .risk-card.danger {
            border-color: #fecaca;
            background: #fef2f2;
        }
        .suggestion-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-top: 0.75rem;
        }
        .suggestion-card {
            border: 1px solid var(--line-soft);
            border-radius: 8px;
            padding: 10px;
            background: #ffffff;
            color: #344054;
            font-size: 0.9rem;
        }
        .adjustment-status {
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            background: #f8fbff;
            padding: 12px 14px;
            margin: 0.9rem 0 0.35rem 0;
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 10px;
            align-items: center;
        }
        .adjustment-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 3px;
        }
        .adjustment-detail {
            color: #475467;
            font-size: 0.94rem;
        }
        .adjustment-pill {
            border-radius: 999px;
            background: #e0ecff;
            color: #1d4ed8;
            font-weight: 700;
            padding: 7px 11px;
            white-space: nowrap;
        }
        .section-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px 18px;
            margin: 0.75rem 0 1rem 0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .module-title {
            color: var(--text);
            font-size: 1.25rem;
            font-weight: 800;
            margin: 0.15rem 0 0.65rem 0;
        }
        div[data-testid="stTabs"] [role="tab"] {
            color: var(--muted);
            font-weight: 650;
        }
        div[data-testid="stTabs"] [aria-selected="true"] {
            color: var(--blue);
        }
        @media (max-width: 1100px) {
            .summary-strip, .prediction-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .suggestion-grid { grid-template-columns: 1fr; }
            .adjustment-status { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def team_select_options(teams: list[str]) -> list[str]:
    return [NO_TEAM_SELECTION] + teams


def team_select_display(team: str) -> str:
    if team == NO_TEAM_SELECTION:
        return "请选择球队"
    return team_display_name(team)


def team_input(label: str, teams: list[str], default_index: int, allow_manual: bool, manual_key: str, select_key: str) -> str:
    if allow_manual:
        return st.text_input(label, value="", key=manual_key).strip()

    options = team_select_options(teams)
    current_value = st.session_state.get(select_key, NO_TEAM_SELECTION)
    if current_value not in options:
        st.session_state[select_key] = NO_TEAM_SELECTION
        current_value = NO_TEAM_SELECTION

    selected = st.selectbox(
        label,
        options,
        index=options.index(current_value),
        format_func=team_select_display,
        key=select_key,
    )
    return "" if selected == NO_TEAM_SELECTION else selected


def default_prediction_date(competition: str) -> date:
    if competition == "WorldCup":
        return date(2026, 7, 3)
    return date(2026, 7, 1)


def default_competition_index(competitions: list[str]) -> int:
    if "WorldCup" in competitions:
        return competitions.index("WorldCup")
    return 0


def selectable_competitions(matches: pd.DataFrame) -> list[str]:
    competitions = set(matches["competition"].astype(str).unique())
    if "WorldCupQualifiers" in competitions:
        competitions.add("WorldCup")
        competitions.discard("WorldCupQualifiers")
    return sorted(competitions)


def competition_display_name(code: str) -> str:
    config = COMPETITIONS.get(code)
    if config is None:
        return code
    if code == "WorldCup":
        return "世界杯（正赛 + 预选赛周期）"
    return config.label


def default_away_index(teams: list[str], selected_competition: str) -> int:
    if selected_competition == "WorldCup" and "Austria" in teams:
        return teams.index("Austria")
    return 1 if len(teams) > 1 else 0


def default_home_index(teams: list[str], selected_competition: str) -> int:
    if selected_competition == "WorldCup" and "Spain" in teams:
        return teams.index("Spain")
    return 0


def html_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def render_summary_strip(items: list[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="summary-item"><div class="summary-label">{html_escape(label)}</div><div class="summary-value">{html_escape(value)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="summary-strip">{cards}</div>', unsafe_allow_html=True)


def render_prediction_grid(items: list[tuple[str, str, bool]]) -> None:
    cards = "".join(prediction_card_html(item) for item in items)
    st.markdown(f'<div class="prediction-grid">{cards}</div>', unsafe_allow_html=True)


def prediction_card_html(item: tuple) -> str:
    label, value, primary, *extra = item
    classes = ["prediction-card"]
    if primary:
        classes.append("primary")
    classes.extend(str(name) for name in extra if name)
    class_attr = " ".join(classes)
    value_html = str(value) if "html-value" in classes else html_escape(value)
    return f'<div class="{class_attr}"><div class="prediction-label">{html_escape(label)}</div><div class="prediction-value">{value_html}</div></div>'


def prediction_grid_html(items: list[tuple]) -> str:
    cards = "".join(prediction_card_html(item) for item in items)
    return f'<div class="prediction-grid">{cards}</div>'


def risk_panel_html(level: str, confidence: float, data_completeness: float) -> str:
    risk_class = "good" if level == "低风险" else "warn" if level == "中风险" else "danger"
    return f"""
    <div class="risk-panel">
      <div class="risk-card {risk_class}">
        <div class="risk-label">风险评级</div>
        <div class="risk-value">{html_escape(level)}</div>
      </div>
      <div class="risk-card">
        <div class="risk-label">综合置信度</div>
        <div class="risk-value">{html_escape(pct(confidence))}</div>
      </div>
      <div class="risk-card">
        <div class="risk-label">数据完整度</div>
        <div class="risk-value">{html_escape(pct(data_completeness))}</div>
      </div>
    </div>
    """


def top_scores_compact_html(scores: list[tuple[str, float]]) -> str:
    rows = "".join(
        f'<div class="top-score-item"><span class="top-score-score">{html_escape(score)}</span><span class="top-score-prob">{html_escape(pct(prob))}</span></div>'
        for score, prob in scores
    )
    return f'<div class="top-score-list">{rows}</div>'


def render_app_header(mode: str) -> None:
    if mode == "单场预测":
        title = "Dixon-Coles 足球预测"
        subtitle = "基于泊松分布与相关性调整的足球概率模型"
        chips = []
    else:
        title = "补录比赛数据"
        subtitle = "把最新完赛结果写入已有 CSV 数据表，刷新后自动参与模型训练"
        chips = []
    chip_html = "".join(
        f'<span class="hero-chip {html_escape(tone)}">{html_escape(label)}</span>'
        for label, tone in chips
    )
    chip_section = f'<div class="hero-chips">{chip_html}</div>' if chip_html else ""
    st.markdown(
        f"""
        <div class="app-hero">
          <h1 class="app-title">{html_escape(title)}</h1>
          <div class="app-subtitle">{html_escape(subtitle)}</div>
          {chip_section}
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_group_summary_from_keys(keys: set[str], data_sources=None) -> str:
    sources = data_sources or discover_data_sources()
    groups = []
    for group in ["世界杯", "英超", "中超"]:
        if any(source.key in keys and source_group(source) == group for source in sources):
            groups.append(group)
    return "+".join(groups) if groups else "未选数据"


def current_source_summary() -> str:
    sources = discover_data_sources()
    available_keys = {source.key for source in sources}
    persisted_keys, has_persisted_sources = load_persisted_source_keys(available_keys)
    active_keys = set()
    for source in sources:
        state_key = f"source_{source.key}"
        if state_key in st.session_state:
            if bool(st.session_state[state_key]):
                active_keys.add(source.key)
        elif has_persisted_sources and source.key in persisted_keys:
            active_keys.add(source.key)
    if not active_keys and not has_persisted_sources:
        active_keys = {source.key for source in sources if source.label in set(default_source_labels())}
    return source_group_summary_from_keys(active_keys, sources)


def render_sidebar_collapsed_summary(summary: str) -> None:
    st.markdown(
        f'<div class="sidebar-collapsed-summary">{html_escape(summary)}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def logo_data_uri() -> str:
    if not LOGO_ASSET_PATH.exists():
        return ""
    encoded = base64.b64encode(LOGO_ASSET_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_sidebar_brand(summary: str) -> None:
    logo_uri = logo_data_uri()
    logo_html = f'<img class="sidebar-logo" src="{logo_uri}" alt="Football logo">' if logo_uri else ""
    st.markdown(
        f"""
        <div class="sidebar-brand">
          {logo_html}
          <div class="sidebar-brand-text">
            <div class="sidebar-brand-title">Dixon-Coles</div>
            <div class="sidebar-brand-subtitle">足球预测系统 · {html_escape(summary)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def manual_adjustment_defaults() -> dict[str, object]:
    return {
        "home_attack_pct": 0.0,
        "home_defense_pct": 0.0,
        "away_attack_pct": 0.0,
        "away_defense_pct": 0.0,
        "home_fitness_pct": 0.0,
        "away_fitness_pct": 0.0,
        "note": "",
    }


def confirmed_manual_adjustments() -> dict[str, object]:
    if "confirmed_manual_adjustments" not in st.session_state:
        st.session_state["confirmed_manual_adjustments"] = manual_adjustment_defaults()
    return st.session_state["confirmed_manual_adjustments"]


def adjustment_from_state(state: dict[str, object]) -> ManualAdjustments:
    return ManualAdjustments(
        home_attack_pct=float(state.get("home_attack_pct", 0.0) or 0.0),
        home_defense_pct=float(state.get("home_defense_pct", 0.0) or 0.0),
        away_attack_pct=float(state.get("away_attack_pct", 0.0) or 0.0),
        away_defense_pct=float(state.get("away_defense_pct", 0.0) or 0.0),
        home_fitness_pct=float(state.get("home_fitness_pct", 0.0) or 0.0),
        away_fitness_pct=float(state.get("away_fitness_pct", 0.0) or 0.0),
    )


def render_adjustment_suggestions() -> None:
    suggestions = [
        ("关键前锋缺阵", "对应球队进攻 -5% 到 -12%"),
        ("主力中卫/门将缺阵", "对应球队防守 -5% 到 -10%"),
        ("大面积轮换", "进攻或体能 -3% 到 -8%"),
        ("高温/长途旅行/休息不足", "体能 -2% 到 -8%"),
        ("强战意或必须争胜", "进攻 +2% 到 +6%，同时留意防守风险"),
        ("大风/暴雨", "双方进攻 -3% 到 -8%，总进球倾向下降"),
        ("东道主/明显主场氛围", "对应球队进攻或体能 +3% 到 +6%"),
        ("保守淘汰赛策略", "双方进攻 -2% 到 -6%，平局概率通常上升"),
    ]
    cards = "".join(
        f'<div class="suggestion-card"><strong>{html_escape(title)}</strong><br>{html_escape(text)}</div>'
        for title, text in suggestions
    )
    st.markdown(f'<div class="suggestion-grid">{cards}</div>', unsafe_allow_html=True)


def adjustment_summary(base_prediction, adjusted_prediction) -> str:
    home_delta = adjusted_prediction.home_goal_expectation - base_prediction.home_goal_expectation
    away_delta = adjusted_prediction.away_goal_expectation - base_prediction.away_goal_expectation
    if abs(home_delta) < 0.005 and abs(away_delta) < 0.005:
        return "当前未应用人工修正"
    return (
        f"人工修正后预期进球："
        f"{adjusted_prediction.home_goal_expectation:.2f} - {adjusted_prediction.away_goal_expectation:.2f}；"
        f"变化：主队 {home_delta:+.2f}，客队 {away_delta:+.2f}"
    )


def render_adjustment_status(base_prediction, adjusted_prediction, note: str, host_side: str) -> None:
    home_delta = adjusted_prediction.home_goal_expectation - base_prediction.home_goal_expectation
    away_delta = adjusted_prediction.away_goal_expectation - base_prediction.away_goal_expectation
    has_change = abs(home_delta) >= 0.005 or abs(away_delta) >= 0.005
    title = "已应用人工修正" if has_change else "未应用人工修正"
    pill = f"主队 {home_delta:+.2f} / 客队 {away_delta:+.2f}" if has_change else "基础模型"
    detail = (
        f"修正后预期进球：{adjusted_prediction.home_goal_expectation:.2f} - {adjusted_prediction.away_goal_expectation:.2f}"
        if has_change
        else "当前预测使用基础 Dixon-Coles 模型结果。"
    )
    extras = []
    if host_side != "无":
        extras.append(f"东道主倾向：{host_side}")
    if note:
        extras.append(f"备注：{note}")
    if extras:
        detail = f"{detail}；{'；'.join(extras)}"
    st.markdown(
        f"""
        <div class="adjustment-status">
          <div>
            <div class="adjustment-title">{html_escape(title)}</div>
            <div class="adjustment-detail">{html_escape(detail)}</div>
          </div>
          <div class="adjustment-pill">{html_escape(pill)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_scope_note(selected_competition: str) -> None:
    if selected_competition == "WorldCup":
        st.caption("模型口径：世界杯可合并正赛与预选赛周期训练；输出为 90 分钟概率，不预测点球或晋级。")
    else:
        st.caption("模型口径：历史比分训练；赔率和人工修正只影响本场预测展示。")


def training_notice(matches: pd.DataFrame, competition: str, prediction_date: date, latest_train_date: date) -> None:
    competition_matches = filter_competition(matches, competition)
    non_ft90 = 0
    if "score_basis" in competition_matches.columns:
        basis = competition_matches["score_basis"].fillna("FT90").replace("", "FT90").astype(str).str.upper()
        non_ft90 = int((basis != "FT90").sum())
    if prediction_date > latest_train_date:
        st.info(f"当前训练数据到 {latest_train_date}，预测 {prediction_date} 属于未来赛前预测。")
    if non_ft90:
        st.warning(f"检测到 {non_ft90} 场非 FT90 数据，已从 Dixon-Coles 训练中排除。请补充 90 分钟比分后再用于训练。")


def history_preview(matches: pd.DataFrame, competition: str, home_team: str | None = None, away_team: str | None = None, limit: int = 20) -> pd.DataFrame:
    columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "stage", "round", "score_basis", "winner", "decided_by_penalties", "notes"]
    available = [column for column in columns if column in matches.columns]
    preview = filter_competition(matches, competition)
    selected_teams = {team for team in [home_team, away_team] if team}
    if selected_teams:
        preview = preview[preview["home_team"].isin(selected_teams) | preview["away_team"].isin(selected_teams)]
    preview = preview[available].tail(limit).copy()
    if "home_team" in preview.columns:
        preview["home_team"] = preview["home_team"].map(team_display_name)
    if "away_team" in preview.columns:
        preview["away_team"] = preview["away_team"].map(team_display_name)
    if "winner" in preview.columns:
        preview["winner"] = preview["winner"].map(lambda team: team_display_name(team) if isinstance(team, str) and team else "")
    if "stage" in preview.columns:
        preview["stage"] = preview["stage"].map(stage_display_name)
    if "round" in preview.columns:
        preview["round"] = preview["round"].map(round_display_name)
    if "score_basis" in preview.columns:
        preview["score_basis"] = preview["score_basis"].map(score_basis_display_name)
    if "decided_by_penalties" in preview.columns:
        preview["decided_by_penalties"] = preview["decided_by_penalties"].map(lambda value: "是" if bool(value) else "否")
    preview = preview.rename(
        columns={
            "date": "日期",
            "home_team": "主队",
            "away_team": "客队",
            "home_goals": "主队进球",
            "away_goals": "客队进球",
            "stage": "阶段",
            "round": "轮次",
            "score_basis": "比分口径",
            "winner": "晋级/胜者",
            "decided_by_penalties": "是否点球",
            "notes": "备注",
        }
    )
    return preview.sort_values("日期", ascending=False)


def stage_display_name(stage: object) -> str:
    labels = {"Group": "小组赛", "Knockout": "淘汰赛", "Qualification": "预选赛"}
    return labels.get(str(stage), str(stage))


def round_display_name(round_name: object) -> str:
    labels = {
        "Group Stage": "小组赛",
        "Round of 32": "32强淘汰赛",
        "Round of 16": "16强淘汰赛",
        "Quarterfinals": "1/4决赛",
        "Semifinals": "半决赛",
        "Third Place": "三四名决赛",
        "Final": "决赛",
        "Qualification": "预选赛",
    }
    return labels.get(str(round_name), str(round_name))


def score_basis_display_name(score_basis: object) -> str:
    if str(score_basis).upper() == "FT90":
        return "90分钟比分（FT90）"
    return str(score_basis)


def top_items(probabilities: dict[str, float], n: int = 5) -> list[tuple[str, float]]:
    return sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:n]


def compact_probability_frame(items: list[tuple[str, float]], label_col: str = "项目") -> pd.DataFrame:
    return pd.DataFrame([{label_col: label, "概率": prob} for label, prob in items])


def total_line_display_items(items: dict[str, float]) -> list[tuple[str, float]]:
    output = []
    for label, probability in items.items():
        side, line = label.split(" ", 1)
        output.append((f"{selection_label(side)} {line}", probability))
    return output


def resolve_odds(csv_odds: pd.DataFrame, competition: str, home_team: str, away_team: str, prediction_date, market: str, manual: dict[str, float], line: float | None = None) -> dict[str, float]:
    csv_values = match_odds(csv_odds, competition, home_team, away_team, prediction_date, market, line=line)
    merged = {key: value for key, value in manual.items() if value is not None and pd.notna(value)}
    merged.update({key: value for key, value in csv_values.items() if pd.notna(value)})
    return merged


def load_local_state(path: Path = LOCAL_STATE_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_local_state(state: dict, path: Path = LOCAL_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_persisted_source_keys(available_keys: set[str], path: Path = LOCAL_STATE_PATH) -> tuple[set[str], bool]:
    state = load_local_state(path)
    if "selected_source_keys" not in state:
        return set(), False
    keys = {str(key) for key in state.get("selected_source_keys", [])}
    return keys & available_keys, True


def save_persisted_source_keys(keys: list[str], path: Path = LOCAL_STATE_PATH) -> None:
    state = load_local_state(path)
    state["selected_source_keys"] = keys
    save_local_state(state, path)


def clear_persisted_source_keys(path: Path = LOCAL_STATE_PATH) -> None:
    state = load_local_state(path)
    state.pop("selected_source_keys", None)
    save_local_state(state, path)


def render_source_selector() -> list[str]:
    st.markdown('<div class="sidebar-section-label">加载数据源</div>', unsafe_allow_html=True)
    selected = []
    selected_keys = []
    defaults = set(default_source_labels())
    data_sources = discover_data_sources()
    source_keys = {source.key for source in data_sources}
    persisted_keys, has_persisted_sources = load_persisted_source_keys(source_keys)
    if st.button("恢复默认数据源", key="reset_source_selection", width="stretch"):
        clear_persisted_source_keys()
        for source in data_sources:
            st.session_state.pop(f"source_{source.key}", None)
        st.rerun()
    selected_group_names = set()
    for source in data_sources:
        state_key = f"source_{source.key}"
        default_checked = source.key in persisted_keys if has_persisted_sources else source.label in defaults
        if bool(st.session_state.get(state_key, default_checked)):
            selected_group_names.add(source_group(source))
    group_order = ["世界杯", "英超", "中超"]
    for group in group_order:
        sources = [source for source in data_sources if source_group(source) == group]
        if not sources:
            continue
        selected_count = sum(
            1
            for source in sources
            if bool(
                st.session_state.get(
                    f"source_{source.key}",
                    (source.key in persisted_keys if has_persisted_sources else source.label in defaults),
                )
            )
        )
        group_label = f"{group} · 已选 {selected_count}" if selected_count else group
        with st.expander(group_label, expanded=(group in selected_group_names)):
            for source in sources:
                match_count = source_match_count(source)
                disabled = not source.path.exists() or match_count == 0
                default_checked = source.key in persisted_keys if has_persisted_sources else source.label in defaults
                checked = st.checkbox(source_display_label(source), value=(default_checked and not disabled), key=f"source_{source.key}", disabled=disabled)
                if checked:
                    selected.append(source.label)
                    selected_keys.append(source.key)
    save_persisted_source_keys(selected_keys)
    return selected


def team_entry_input(label: str, candidates: list[str], manual: bool, prefix: str) -> tuple[str, str]:
    if manual:
        english = st.text_input(f"{label}英文标准名", key=f"{prefix}_english").strip()
        zh = st.text_input(f"{label}中文名", key=f"{prefix}_zh").strip()
        return english, zh
    selected = st.selectbox(label, candidates, format_func=team_display_name, key=f"{prefix}_select")
    return selected, ""


def match_entry_success_message(home_team: str, away_team: str, home_goals: int, away_goals: int, source_label: str) -> str:
    return (
        f"{team_display_name(home_team)} vs {team_display_name(away_team)} "
        f"{home_goals}:{away_goals} 数据已保存到 {source_label}"
    )


def render_match_entry_form(data_sources) -> None:
    if not data_sources:
        st.info("当前没有可补录的数据表。请先在 data/worldcup、data/epl 或 data/csl 中放入 CSV。")
        return

    labels = [source.label for source in data_sources]
    label_map = source_by_label()
    selected_label = st.selectbox("目标数据表", labels, format_func=lambda label: label)
    source = label_map[selected_label]
    defaults = default_entry_values(source)
    existing_teams = target_teams(source)
    candidates = sorted(set(existing_teams) | set(all_team_names_zh().keys()))

    st.caption(f"补录会写入：{source.label}。CSV 内部保存英文标准名，页面显示中文名。")
    manual_team_mode = st.checkbox("手动新增球队 / 队名", value=False)

    with st.form("match_entry_form"):
        row1 = st.columns([0.9, 0.8, 0.8, 0.7])
        with row1[0]:
            match_date = st.date_input("比赛日期", value=date.today())
        with row1[1]:
            competition = st.text_input("赛事代码", value=str(defaults["competition"]))
        with row1[2]:
            season = st.text_input("赛季", value=str(defaults["season"]))
        with row1[3]:
            score_basis = st.selectbox("比分口径", ["FT90", "FT", "full_time"], index=0)

        team_col1, score_col1, score_col2, team_col2 = st.columns([1.3, 0.7, 0.7, 1.3])
        with team_col1:
            home_team, home_zh = team_entry_input("主队", candidates, manual_team_mode, "entry_home")
        with score_col1:
            home_goals = st.number_input("主队90分钟进球", min_value=0, value=0, step=1)
        with score_col2:
            away_goals = st.number_input("客队90分钟进球", min_value=0, value=0, step=1)
        with team_col2:
            away_team, away_zh = team_entry_input("客队", candidates, manual_team_mode, "entry_away")

        row3 = st.columns([0.7, 0.8, 1, 0.8])
        with row3[0]:
            neutral_site = st.checkbox("中立场", value=bool(defaults["neutral_site"]))
        with row3[1]:
            stage = st.selectbox("阶段", ["Group", "Knockout", "Qualification", "League", ""], index=["Group", "Knockout", "Qualification", "League", ""].index(str(defaults.get("stage", ""))))
        with row3[2]:
            round_name = st.text_input("轮次", value=str(defaults.get("round", "")))
        with row3[3]:
            decided_by_penalties = st.checkbox("是否点球决胜", value=False)

        winner_options = ["", home_team, away_team]
        winner = st.selectbox("晋级/胜者（可选，仅记录展示）", winner_options, format_func=lambda team: "不填写" if not team else team_display_name(team))
        notes = st.text_area("备注", placeholder="例如：Germany advanced 5-4 on penalties；或记录天气、红牌、阵容等。")

        submitted = st.form_submit_button("保存补录比赛")

    success_message = st.session_state.get("match_entry_success")
    if success_message:
        st.success(success_message)

    if not submitted:
        return

    entry = MatchEntry(
        competition=competition.strip(),
        season=season.strip(),
        date=match_date.strftime("%Y-%m-%d"),
        home_team=home_team.strip(),
        away_team=away_team.strip(),
        home_goals=int(home_goals),
        away_goals=int(away_goals),
        neutral_site=neutral_site,
        stage=stage,
        round=round_name.strip(),
        score_basis=score_basis,
        decided_by_penalties=decided_by_penalties,
        winner=winner.strip(),
        notes=notes.strip(),
    )
    try:
        backup_path = append_match_entry(source, entry)
        if manual_team_mode:
            save_custom_team_name(home_team, home_zh)
            save_custom_team_name(away_team, away_zh)
        clear_catalog_cache()
        cached_train_model.clear()
        backup_note = f"已备份到 {backup_path}" if backup_path else "目标表原先不存在，已新建"
        st.session_state["match_entry_success"] = (
            f"{match_entry_success_message(home_team, away_team, int(home_goals), int(away_goals), source.label)}。{backup_note}。"
        )
        st.rerun()
    except Exception as exc:
        st.error(f"补录失败：{exc}")


def render_data_entry_page() -> None:
    with st.container(border=True):
        st.markdown('<div class="module-title">补录比赛数据</div>', unsafe_allow_html=True)
        st.caption("CSV 内部保存英文标准名，页面显示中文名；点球和晋级信息只记录展示。")
        render_match_entry_form(discover_data_sources())


def main() -> None:
    st.set_page_config(page_title="Dixon-Coles 足球预测", page_icon=str(LOGO_ASSET_PATH), layout="wide")
    inject_styles()

    with st.sidebar:
        sidebar_source_summary = current_source_summary()
        render_sidebar_collapsed_summary(sidebar_source_summary)
        render_sidebar_brand(sidebar_source_summary)
        st.markdown('<div class="sidebar-section-label">功能入口</div>', unsafe_allow_html=True)
        page_mode = st.radio("当前页面", ["单场预测", "补录比赛数据"], label_visibility="collapsed")

    render_app_header(page_mode)

    if page_mode == "补录比赛数据":
        render_data_entry_page()
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-section-label">数据与参数</div>', unsafe_allow_html=True)
        selected_sources = render_source_selector()
        qualifier_fast_mode = st.checkbox(
            "训练快速模式",
            value=True,
            help="世界杯：只加入当前双方相关的预选赛。英超/中超：数据过多时保留最近比赛。用于避免大数据量训练卡顿或不收敛。",
        )
        uploaded_files = st.file_uploader("上传比赛 CSV", type=["csv"], accept_multiple_files=True)
        uploaded_odds_files = st.file_uploader("上传赔率 CSV", type=["csv"], accept_multiple_files=True)
        uploaded_strength_files = st.file_uploader("上传球队强度 CSV", type=["csv"], accept_multiple_files=True, help="字段：team, confederation, rating。上传值会覆盖内置先验。")
        half_life_days = st.number_input("时间衰减半衰期（Half-life days）", min_value=0.0, value=365.0, step=30.0)
        max_goals = st.slider("比分矩阵最大进球", 5, 12, 8)
        allow_manual_teams = st.checkbox("手动输入队伍", value=False)
        use_strength_correction = st.checkbox("跨赛区强度校正", value=True, help="仅世界杯模式生效。双方赛区不同时，按本地强度评分小幅修正预期进球。")
        strength_intensity = st.slider("强度校正力度", 0.0, 1.0, 0.60, 0.05)
        market_weight = st.slider("赔率融合权重（Market blend weight）", 0.0, 1.0, 0.30, 0.05)

    catalog_matches, loaded_sources = load_catalog_sources(selected_sources)
    frames = [catalog_matches]
    frames.extend(load_uploaded(uploaded_files))
    matches = combine_match_frames(frames)
    csv_odds = load_uploaded_odds(uploaded_odds_files)
    uploaded_strength_frames = load_project_strength_frames() + load_uploaded_strength(uploaded_strength_files)

    if matches.empty:
        st.info("请先加载样例数据或上传比赛 CSV。CSV 至少需要 competition, season, date, home_team, away_team, home_goals, away_goals, neutral_site。")
        return

    with st.container(border=True):
        st.markdown('<div class="module-title">赛事与比赛设置</div>', unsafe_allow_html=True)
        competitions = selectable_competitions(matches)
        current_competition = st.session_state.get("selected_competition", None)
        if current_competition not in competitions:
            st.session_state["selected_competition"] = competitions[default_competition_index(competitions)]
        selected_competition = st.selectbox(
            "赛事",
            competitions,
            index=competitions.index(st.session_state["selected_competition"]),
            format_func=competition_display_name,
            key="selected_competition",
        )
        model_scope_note(selected_competition)
        teams = teams_for_competition(matches, selected_competition)
        if len(teams) < 2:
            st.warning("当前赛事至少需要两支球队。")
            return
        left, right, date_col, neutral_col = st.columns([1, 1, 0.9, 0.7])
        with left:
            home_team = team_input("主队 / 队伍 A", teams, default_home_index(teams, selected_competition), allow_manual_teams, "manual_home", "selected_home")
        with right:
            away_default = default_away_index(teams, selected_competition)
            away_team = team_input("客队 / 队伍 B", teams, away_default, allow_manual_teams, "manual_away", "selected_away")
        with date_col:
            prediction_date = st.date_input("预测日期", value=default_prediction_date(selected_competition))
        with neutral_col:
            competition_matches = filter_competition(matches, selected_competition)
            default_neutral = bool(competition_matches["neutral_site"].mode().iloc[0])
            neutral_site = st.checkbox("中立场", value=default_neutral)
        host_side = "无"
        if selected_competition == "WorldCup" and neutral_site:
            host_side = st.segmented_control("东道主/主场氛围", ["无", "主队", "客队"], default="无")
            st.caption("中立场默认不使用普通主场优势；如果一方有明显东道主或现场氛围，可在这里做小幅赛前修正。")
        elif not neutral_site:
            st.caption("非中立场时，主队会自动使用 Dixon-Coles 的普通主场优势。")

    competition_label = competition_display_name(selected_competition)

    if not home_team or not away_team:
        st.info("请选择主队和客队后开始预测。切换数据源时，已选球队如果仍存在会自动保留。")
        return

    if home_team == away_team:
        st.warning("请选择两支不同球队。")
        return

    training_matches = optimize_training_matches(matches, selected_competition, home_team, away_team, qualifier_fast_mode)
    if qualifier_fast_mode and selected_competition == "WorldCup":
        original_qualifiers = int((matches["competition"].astype(str) == "WorldCupQualifiers").sum())
        used_qualifiers = int((training_matches["competition"].astype(str) == "WorldCupQualifiers").sum())
        if original_qualifiers and used_qualifiers < original_qualifiers:
            st.caption(f"训练快速模式：本次只加入 {used_qualifiers}/{original_qualifiers} 场与当前双方相关的预选赛，提升训练速度。")
    elif qualifier_fast_mode and selected_competition in {"EPL", "CSL"}:
        original_rows = len(filter_competition(matches, selected_competition))
        used_rows = len(filter_competition(training_matches, selected_competition))
        if used_rows < original_rows:
            st.caption(f"训练快速模式：本次使用最近 {used_rows}/{original_rows} 场联赛比赛。")
    elif selected_competition == "WorldCup" and int((matches["competition"].astype(str) == "WorldCupQualifiers").sum()) > 300:
        st.warning("当前包含大量世界杯预选赛数据，关闭快速模式可能训练很久或不收敛；建议开启“训练快速模式”。")

    try:
        with st.spinner("正在训练模型，首次加载较慢；后续相同数据和参数会使用缓存。"):
            trained = cached_train_model(matches_cache_payload(training_matches), selected_competition, half_life_days)
    except Exception as exc:
        qualifier_count = int((matches["competition"].astype(str) == "WorldCupQualifiers").sum()) if "competition" in matches.columns else 0
        if not qualifier_fast_mode:
            fallback_matches = optimize_training_matches(matches, selected_competition, home_team, away_team, True)
            try:
                with st.spinner("全量训练失败，正在自动改用快速模式重试。"):
                    trained = cached_train_model(matches_cache_payload(fallback_matches), selected_competition, half_life_days)
                training_matches = fallback_matches
                if selected_competition == "WorldCup" and qualifier_count:
                    used_qualifiers = int((training_matches["competition"].astype(str) == "WorldCupQualifiers").sum())
                    st.warning(f"全量训练失败，已自动改用快速模式：本次使用 {used_qualifiers}/{qualifier_count} 场相关预选赛。原始错误：{exc}")
                else:
                    st.warning(f"全量训练失败，已自动改用快速模式。原始错误：{exc}")
            except Exception as fallback_exc:
                st.error(f"模型训练失败：{fallback_exc}")
                return
        else:
            st.error(f"模型训练失败：{exc}")
            return

    if hasattr(trained.model, "fit_converged_") and not trained.model.fit_converged_:
        st.warning(f"模型已生成预测，但优化器达到评估上限，当前使用近似收敛参数。提示：{trained.model.fit_message_}")

    if home_team not in trained.model.teams_ or away_team not in trained.model.teams_:
        missing = [team for team in [home_team, away_team] if team not in trained.model.teams_]
        st.warning(f"缺少该队历史比赛，需补充数据后预测：{', '.join(missing)}")
        st.caption(f"当前 {competition_display_name(selected_competition)} 已有球队：{', '.join(team_display_name(team) for team in trained.model.teams_)}")
        return

    try:
        base_prediction = trained.model.predict(home_team, away_team, max_goals=max_goals, neutral_site=neutral_site)
    except Exception as exc:
        st.error(f"预测失败：{exc}")
        return

    confirmed_state = confirmed_manual_adjustments()
    effective_state = confirmed_state.copy()
    if host_side == "主队":
        effective_state["home_attack_pct"] = float(effective_state.get("home_attack_pct", 0.0)) + 4.0
        effective_state["home_fitness_pct"] = float(effective_state.get("home_fitness_pct", 0.0)) + 3.0
    elif host_side == "客队":
        effective_state["away_attack_pct"] = float(effective_state.get("away_attack_pct", 0.0)) + 4.0
        effective_state["away_fitness_pct"] = float(effective_state.get("away_fitness_pct", 0.0)) + 3.0
    adjusted_prediction = apply_manual_adjustments(
        base_prediction,
        trained.model.rho_,
        adjustment_from_state(effective_state),
        max_goals,
    )
    adjustment_note = str(confirmed_state.get("note", "")).strip()

    with st.expander("高级人工修正", expanded=False):
        st.caption("修正只作用于本场预测的预期进球，不改训练数据和球队参数。填写后需要点击确认才会生效。备注只记录判断依据，不直接参与计算。")
        with st.form("manual_adjustment_form"):
            a1, a2, a3 = st.columns(3)
            with a1:
                home_attack_adj = st.number_input("主队进攻修正 %", value=float(confirmed_state["home_attack_pct"]), step=1.0)
                away_attack_adj = st.number_input("客队进攻修正 %", value=float(confirmed_state["away_attack_pct"]), step=1.0)
            with a2:
                home_defense_adj = st.number_input("主队防守修正 %", value=float(confirmed_state["home_defense_pct"]), step=1.0, help="正数表示防守增强，会降低对手预期进球。")
                away_defense_adj = st.number_input("客队防守修正 %", value=float(confirmed_state["away_defense_pct"]), step=1.0, help="正数表示防守增强，会降低对手预期进球。")
            with a3:
                home_fitness_adj = st.number_input("主队体能/赛程修正 %", value=float(confirmed_state["home_fitness_pct"]), step=1.0)
                away_fitness_adj = st.number_input("客队体能/赛程修正 %", value=float(confirmed_state["away_fitness_pct"]), step=1.0)
            note = st.text_area("人工备注", value=str(confirmed_state.get("note", "")), placeholder="例如：主力前锋缺阵、连续客场、轮换、战意更强、天气影响等。")
            form_actions = st.columns([0.7, 0.7, 2])
            with form_actions[0]:
                apply_adjustment = st.form_submit_button("确认应用人工修正")
            with form_actions[1]:
                clear_adjustment = st.form_submit_button("清空人工修正")
        render_adjustment_suggestions()
        if apply_adjustment:
            st.session_state["confirmed_manual_adjustments"] = {
                "home_attack_pct": home_attack_adj,
                "home_defense_pct": home_defense_adj,
                "away_attack_pct": away_attack_adj,
                "away_defense_pct": away_defense_adj,
                "home_fitness_pct": home_fitness_adj,
                "away_fitness_pct": away_fitness_adj,
                "note": note,
            }
            st.rerun()
        if clear_adjustment:
            st.session_state["confirmed_manual_adjustments"] = manual_adjustment_defaults()
            st.rerun()
        render_adjustment_status(base_prediction, adjusted_prediction, adjustment_note, host_side)

    strength_defaults = default_strength_records(teams_for_competition(matches, selected_competition)) if selected_competition == "WorldCup" else pd.DataFrame()
    strength_frame = combine_strength_frames(strength_defaults, uploaded_strength_frames)
    strength_correction = apply_cross_confederation_strength_correction(
        adjusted_prediction,
        strength_lookup(strength_frame),
        enabled=selected_competition == "WorldCup" and use_strength_correction,
        intensity=strength_intensity,
        rho=trained.model.rho_,
    )

    prediction = strength_correction.prediction

    result_probs = prediction.result_probabilities()
    total_probs = prediction.over_under(2.5)
    btts_probs = prediction.both_teams_to_score()
    top_score_html = top_scores_compact_html(prediction.top_scores(3))
    has_manual_change = adjusted_prediction.home_goal_expectation != base_prediction.home_goal_expectation or adjusted_prediction.away_goal_expectation != base_prediction.away_goal_expectation
    has_xg = any(column in training_matches.columns and training_matches[column].notna().any() for column in ["home_xg", "away_xg"])
    risk = assess_match_risk(prediction, training_matches, home_team, away_team)
    scope_items = model_scope_summary(
        selected_competition,
        uses_market=False,
        uses_manual_adjustment=has_manual_change or host_side != "无",
        has_xg=has_xg,
    )
    core_items = [
        ("预期进球（主队 - 客队）", f"{prediction.home_goal_expectation:.2f} - {prediction.away_goal_expectation:.2f}", True),
        ("主胜概率（90分钟）", pct(result_probs["home_win"]), True, "good"),
        ("平局概率（90分钟）", pct(result_probs["draw"]), False, "warn"),
        ("客胜概率（90分钟）", pct(result_probs["away_win"]), True, "danger"),
        ("大 2.5 球", pct(total_probs["over"]), False),
        ("小 2.5 球", pct(total_probs["under"]), False),
        ("双方进球（BTTS）", pct(btts_probs["yes"]), False),
        ("最可能比分", top_score_html, True, "top-score", "html-value"),
    ]
    host_label = f" · 东道主倾向：{host_side}" if selected_competition == "WorldCup" and neutral_site and host_side != "无" else ""
    st.markdown(
        f"""
        <div class="match-summary">
          <h3>{html_escape(matchup_display(home_team, away_team))}</h3>
          <div class="muted">
            {html_escape(competition_label)} · 预测日期 {html_escape(str(prediction_date))} · 已训练 {trained.matches_used} 场 · 数据至 {trained.last_match_date.date()} · {'中立场' if neutral_site else '主客场'}{html_escape(host_label)}
          </div>
          {prediction_grid_html(core_items)}
          {risk_panel_html(risk.level, risk.confidence, risk.data_completeness)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("当前模型口径与风险说明", expanded=False):
        st.markdown("**模型版本：Dixon-Coles Baseline + Elo/xG 特征层已就绪**")
        for item in scope_items:
            st.markdown(f"- {item}")
        st.markdown("**风险原因**")
        for reason in risk.reasons:
            st.markdown(f"- {reason}")
    if selected_competition == "WorldCup":
        if strength_correction.applied:
            st.caption(
                "跨赛区强度校正已应用："
                f"{team_display_name(home_team)} {strength_correction.home_confederation} "
                f"{strength_correction.home_rating:.0f} vs "
                f"{team_display_name(away_team)} {strength_correction.away_confederation} "
                f"{strength_correction.away_rating:.0f}；"
                f"预期进球对数修正 {strength_correction.log_adjustment:+.3f}。"
            )
        elif use_strength_correction:
            st.caption(f"跨赛区强度校正未应用：{strength_correction.reason}。同赛区或缺少评分时不修正。")
    training_notice(training_matches, selected_competition, prediction_date, trained.last_match_date.date())

    with st.container(border=True):
        st.markdown('<div class="module-title">双方历史数据</div>', unsafe_allow_html=True)
        st.caption("根据当前选择的主队/客队过滤已加载数据；包含任一球队的比赛都会显示。")
        selected_history = history_preview(matches, selected_competition, home_team, away_team, limit=30)
        if selected_history.empty:
            st.info("当前加载数据中没有找到这两支球队的历史比赛。")
        else:
            st.dataframe(selected_history, width="stretch", hide_index=True, height=320)

    with st.container(border=True):
        st.markdown('<div class="module-title">模型预测详情</div>', unsafe_allow_html=True)
        score_rows = [{"比分": score, "概率": prob} for score, prob in prediction.top_scores(10)]
        score_frame = pd.DataFrame(score_rows)
        heatmap = pd.DataFrame(
            prediction.score_matrix,
            index=[f"{i}球" for i in range(prediction.score_matrix.shape[0])],
            columns=[f"{i}球" for i in range(prediction.score_matrix.shape[1])],
        )
        detail_tabs = st.tabs(["比分", "进球数", "让球", "半全场"])
        with detail_tabs[0]:
            score_col, heat_col = st.columns([0.7, 1.3])
            with score_col:
                st.dataframe(score_frame.style.format({"概率": pct}), width="stretch", hide_index=True)
            with heat_col:
                st.dataframe(styled_probability_matrix(heatmap), width="stretch")
        with detail_tabs[1]:
            total_dist = total_goals_distribution(prediction)
            common_totals = over_under_lines(prediction)
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**总进球数分布**")
                st.dataframe(compact_probability_frame(list(total_dist.items()), "进球数").style.format({"概率": pct}), width="stretch", hide_index=True)
            with g2:
                st.markdown("**大小球 1.5 / 2.5 / 3.5**")
                st.dataframe(compact_probability_frame(total_line_display_items(common_totals), "盘口").style.format({"概率": pct}), width="stretch", hide_index=True)
        with detail_tabs[2]:
            h1, h2 = st.columns(2)
            with h1:
                handicap_line = st.select_slider("让球胜平负盘口（主队）", options=[-2, -1, 0, 1, 2], value=0, key="detail_euro_handicap")
                european_handicap = european_handicap_market(prediction, int(handicap_line))
                st.dataframe(compact_probability_frame([(selection_label(k), v) for k, v in european_handicap.items()], "选项").style.format({"概率": pct}), width="stretch", hide_index=True)
            with h2:
                asian_quick_line = st.select_slider("亚洲让球盘口（主队）", options=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], value=0.0, key="detail_asian_handicap")
                asian_quick = asian_handicap_market(prediction, asian_quick_line)
                st.dataframe(compact_probability_frame([(selection_label(k), v) for k, v in asian_quick.items()], "选项").style.format({"概率": pct}), width="stretch", hide_index=True)
        with detail_tabs[3]:
            half_full = top_items(half_full_time_market(prediction), 9)
            st.dataframe(compact_probability_frame(half_full, "半全场").style.format({"概率": pct}), width="stretch", hide_index=True)

    with st.container(border=True):
        st.markdown('<div class="module-title">赔率与价值</div>', unsafe_allow_html=True)
        st.caption("十进制赔率；赔率只用于市场对比和融合概率。")
        odds_tabs = st.tabs(["胜平负", "大小球", "双方进球", "让球"])
        with odds_tabs[0]:
            o1, o2, o3 = st.columns(3)
            with o1:
                home_odds = parse_decimal_input(st.text_input("主胜", placeholder="输入十进制赔率", key="odds_home_text"))
            with o2:
                draw_odds = parse_decimal_input(st.text_input("平局", placeholder="输入十进制赔率", key="odds_draw_text"))
            with o3:
                away_odds = parse_decimal_input(st.text_input("客胜", placeholder="输入十进制赔率", key="odds_away_text"))
        with odds_tabs[1]:
            o1, o2, o3 = st.columns(3)
            with o1:
                total_line = st.select_slider("大小球盘口", options=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], value=2.5, key="odds_total_line")
            with o2:
                over_odds = parse_decimal_input(st.text_input("大球赔率", placeholder="输入十进制赔率", key="odds_over_text"))
            with o3:
                under_odds = parse_decimal_input(st.text_input("小球赔率", placeholder="输入十进制赔率", key="odds_under_text"))
        with odds_tabs[2]:
            o1, o2 = st.columns(2)
            with o1:
                btts_yes_odds = parse_decimal_input(st.text_input("是", placeholder="输入十进制赔率", key="odds_btts_yes_text"))
            with o2:
                btts_no_odds = parse_decimal_input(st.text_input("否", placeholder="输入十进制赔率", key="odds_btts_no_text"))
        with odds_tabs[3]:
            o1, o2, o3, o4 = st.columns(4)
            with o1:
                asian_line = st.number_input("亚洲让球主队盘口", value=0.0, step=0.5, key="odds_asian_line")
            with o2:
                asian_home_odds = parse_decimal_input(st.text_input("亚洲让球主队赔率", placeholder="输入十进制赔率", key="odds_asian_home_text"))
            with o3:
                asian_away_odds = parse_decimal_input(st.text_input("亚洲让球客队赔率", placeholder="输入十进制赔率", key="odds_asian_away_text"))
            with o4:
                euro_handicap = st.number_input("让球胜平负主队调整球数", value=0, step=1, key="odds_euro_handicap")

        model_1x2 = result_market(prediction)
        odds_1x2 = resolve_odds(csv_odds, selected_competition, home_team, away_team, prediction_date, "1X2", {"Home": home_odds, "Draw": draw_odds, "Away": away_odds})
        market_probs_1x2 = remove_vig(odds_1x2)
        if market_probs_1x2:
            market_risk = assess_match_risk(prediction, training_matches, home_team, away_team, market_probs_1x2)
            st.caption(
                f"市场分歧：{pct(market_risk.model_disagreement)}；"
                f"纳入当前胜平负赔率后风险评级参考为 {market_risk.level}。"
            )
        rows_blended = []
        rows_blended.extend(value_rows_with_blend("胜平负（1X2）", model_1x2, odds_1x2, market_weight))
        rows_blended.extend(value_rows_with_blend("大小球（Over/Under）", totals_market(prediction, total_line), {"Over": over_odds, "Under": under_odds}, market_weight, line=total_line))
        rows_blended.extend(value_rows_with_blend("双方进球（BTTS）", btts_market(prediction), {"Yes": btts_yes_odds, "No": btts_no_odds}, market_weight))
        ah_probs = asian_handicap_market(prediction, asian_line)
        rows_blended.extend(value_rows_with_blend("亚洲让球（Asian Handicap）", {"Home": ah_probs["Home"], "Away": ah_probs["Away"]}, {"Home": asian_home_odds, "Away": asian_away_odds}, market_weight, line=asian_line))
        eh_probs = european_handicap_market(prediction, int(euro_handicap))
        rows_blended.extend(value_rows_with_blend("让球胜平负（European Handicap）", eh_probs, {}, market_weight, line=float(euro_handicap)))
        table = blended_market_table(rows_blended)
        value_filter = st.segmented_control("价值筛选", ["全部", "正 EV", "已填赔率"], default="全部")
        filtered_table = filter_value_table(table, value_filter)
        st.caption("纯模型概率不受赔率影响；融合概率按当前赔率去水概率与模型概率加权。")
        st.dataframe(format_blended_table(filtered_table), width="stretch", hide_index=True, height=360)
        ah_probs = asian_handicap_market(prediction, asian_line)
        if ah_probs.get("Push", 0.0) > 0:
            st.caption(f"亚洲让球走盘概率：{pct(ah_probs['Push'])}，EV 表按扣除走盘后的条件概率展示。")

    with st.expander("数据与模型细节", expanded=False):
        model_parameter_count = trained.teams_used * 2 + 2
        st.caption(
            f"训练比赛 {trained.matches_used} 场 · 球队 {trained.teams_used} 支 · "
            f"加权场次 {trained.weighted_matches:.1f} · "
            f"模型参数约 {model_parameter_count} 个 · "
            f"训练日期 {trained.first_match_date.date()} 至 {trained.last_match_date.date()} · "
            f"90分钟比分（FT90），排除 {trained.excluded_non_ft90} 场"
        )
        if selected_competition == "WorldCup":
            st.caption("权重口径：世界杯正赛 1.00，淘汰赛 1.05，世界杯预选赛 0.45。跨赛区强度校正使用本地评分先验或上传 CSV；xG、球星变量和晋级概率仍未单独建模。")
        if loaded_sources:
            st.caption(f"当前基础模型数据来源：{', '.join(loaded_sources)}")
        if selected_competition == "WorldCup" and not strength_frame.empty:
            st.markdown("**当前两队强度评分**")
            strength_preview = strength_frame[strength_frame["team"].isin([home_team, away_team])].copy()
            if not strength_preview.empty:
                strength_preview.insert(1, "球队中文名", strength_preview["team"].map(team_display_name))
                strength_preview = strength_preview.rename(columns={"team": "球队英文名", "confederation": "赛区", "rating": "强度评分", "source": "来源"})
                st.dataframe(strength_preview, width="stretch", hide_index=True)
        parameters = trained.model.parameters()
        parameters.insert(1, "球队中文名", parameters["team"].map(team_display_name))
        parameters = parameters.rename(columns={"team": "球队英文名", "attack": "进攻强度", "defense": "防守强度"})
        st.dataframe(parameters, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
