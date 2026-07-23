from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .dixon_coles import Prediction


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    confidence: float
    data_completeness: float
    model_disagreement: float
    reasons: list[str]


def assess_match_risk(
    prediction: Prediction,
    training_matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    market_probabilities: dict[str, float] | None = None,
) -> RiskAssessment:
    result_probs = prediction.result_probabilities()
    model_probs = {
        "Home": result_probs["home_win"],
        "Draw": result_probs["draw"],
        "Away": result_probs["away_win"],
    }
    top_probability = max(model_probs.values())
    data_completeness = data_completeness_score(training_matches, home_team, away_team)
    disagreement = market_disagreement(model_probs, market_probabilities or {})
    reasons = []

    if top_probability < 0.42:
        reasons.append("胜平负概率接近，比赛本身不确定性高")
    if data_completeness < 0.55:
        reasons.append("当前两队历史样本偏少")
    if disagreement > 0.18:
        reasons.append("模型概率与市场概率分歧较大")
    if not reasons:
        reasons.append("主要概率、样本量和市场分歧处于可接受范围")

    risk_score = 0.45 * (1.0 - top_probability) + 0.35 * (1.0 - data_completeness) + 0.20 * min(1.0, disagreement / 0.35)
    if risk_score < 0.32:
        level = "低风险"
    elif risk_score < 0.50:
        level = "中风险"
    else:
        level = "高风险"
    confidence = max(0.0, min(1.0, 1.0 - risk_score))
    return RiskAssessment(level, confidence, data_completeness, disagreement, reasons)


def data_completeness_score(matches: pd.DataFrame, home_team: str, away_team: str) -> float:
    if matches.empty:
        return 0.0
    team_rows = matches[(matches["home_team"].isin([home_team, away_team])) | (matches["away_team"].isin([home_team, away_team]))]
    home_count = int(((matches["home_team"] == home_team) | (matches["away_team"] == home_team)).sum())
    away_count = int(((matches["home_team"] == away_team) | (matches["away_team"] == away_team)).sum())
    count_score = min(1.0, min(home_count, away_count) / 10.0)
    recency_score = 0.0
    if not team_rows.empty:
        dates = pd.to_datetime(team_rows["date"], errors="coerce").dropna()
        if not dates.empty:
            latest = pd.to_datetime(matches["date"], errors="coerce").max()
            age_days = max(0, int((latest - dates.max()).days))
            recency_score = max(0.0, 1.0 - age_days / 730.0)
    return float(0.75 * count_score + 0.25 * recency_score)


def market_disagreement(model_probs: dict[str, float], market_probs: dict[str, float]) -> float:
    if not market_probs:
        return 0.0
    common = set(model_probs) & set(market_probs)
    if not common:
        return 0.0
    return max(abs(float(model_probs[key]) - float(market_probs[key])) for key in common)


def model_scope_summary(competition: str, uses_market: bool, uses_manual_adjustment: bool, has_xg: bool) -> list[str]:
    scope = ["预测口径：90分钟常规时间；点球和晋级不作为当前输出目标"]
    if competition == "WorldCup":
        scope.append("世界杯模式：正赛和预选赛可合并训练，但赛事权重与跨赛区修正分开处理")
    else:
        scope.append("联赛模式：按当前赛事单独训练，不与国家队赛事混训")
    scope.append("基础模型：Dixon-Coles 进球模型；赔率不会反向写入训练数据")
    if uses_market:
        scope.append("赔率：只用于市场比较、融合概率和 EV")
    if uses_manual_adjustment:
        scope.append("人工修正：只调整本场预期进球，不改变历史球队参数")
    if not has_xg:
        scope.append("xG：当前数据缺少真实 xG，相关特征保持缺失")
    return scope
