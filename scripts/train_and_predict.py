#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from football_dc import DixonColesModel


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Dixon-Coles model and predict one fixture.")
    parser.add_argument("--matches", required=True, help="CSV file with date, home_team, away_team, home_goals, away_goals.")
    parser.add_argument("--home", required=True, help="Home team name.")
    parser.add_argument("--away", required=True, help="Away team name.")
    parser.add_argument("--half-life-days", type=float, default=365.0, help="Time-decay half life. Use 0 to disable.")
    parser.add_argument("--max-goals", type=int, default=8, help="Maximum goals in score matrix.")
    args = parser.parse_args()

    matches_path = Path(args.matches)
    matches = pd.read_csv(matches_path)

    model = DixonColesModel(half_life_days=args.half_life_days).fit(matches)
    prediction = model.predict(args.home, args.away, max_goals=args.max_goals)

    print(f"{args.home} vs {args.away}")
    print(f"Expected goals: {prediction.home_goal_expectation:.2f} - {prediction.away_goal_expectation:.2f}")
    print()

    result_probs = prediction.result_probabilities()
    print("1X2")
    print(f"Home win: {pct(result_probs['home_win'])}")
    print(f"Draw:     {pct(result_probs['draw'])}")
    print(f"Away win: {pct(result_probs['away_win'])}")
    print()

    over_under = prediction.over_under(2.5)
    print("Total goals 2.5")
    print(f"Over:  {pct(over_under['over'])}")
    print(f"Under: {pct(over_under['under'])}")
    print()

    btts = prediction.both_teams_to_score()
    print("Both teams to score")
    print(f"Yes: {pct(btts['yes'])}")
    print(f"No:  {pct(btts['no'])}")
    print()

    print("Top scores")
    for score, prob in prediction.top_scores(10):
        print(f"{score}: {pct(prob)}")


if __name__ == "__main__":
    main()
